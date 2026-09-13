# A part of NonVisual Desktop Access (NVDA)
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Compare the standard and enhanced Persian frontends using local Piper assets."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
import logging
from pathlib import Path
import platform
import sys
import time
import traceback
from unittest.mock import patch
import wave

from .backend import loadPiperVoice, localVoicePaths


def _defaultCases() -> list[tuple[str, str]]:
	return [
		("word", "دستیار"),
		("word_period", "دستیار."),
		("sentence", "این دستیار به من کمک می‌کند."),
		*((f"letter_{ord(letter):04x}", letter) for letter in "دستیار"),
		*(
			(f"letter_name_{index}", name)
			for index, name in enumerate(("دال", "سین", "تِ", "یِ", "الف", "رِ"), 1)
		),
		# These are listening comparisons, not authoritative pronunciation targets.
		("candidate_stress_only", "[[dastjˈɑr]]"),
		("candidate_alternate_ipa", "[[dæstjˈɒːɾ]]"),
	]


class _WarningLog(logging.Handler):
	def __init__(self):
		super().__init__(logging.WARNING)
		self.messages: list[str] = []

	def emit(self, record):
		self.messages.append(f"{record.name}: {record.getMessage()}")


def _acousticMetrics(samples, sampleRate: int) -> dict:
	"""Describe high-frequency energy in normalized samples; this is not an intelligibility test."""
	import numpy as np

	frameSize = min(int(sampleRate * 0.04), len(samples))
	result = {
		"frame_seconds": 0.04,
		"hop_seconds": 0.01,
		"activity_rms_full_scale": 0.01,
		"high_frequency_cutoff_hz": 4000,
		"active_frames": 0,
		"mean_high_frequency_energy_fraction": None,
		"median_high_frequency_energy_fraction": None,
	}
	if frameSize < 3 or sampleRate <= 8000:
		return result
	window = np.hanning(frameSize)
	highBins = np.fft.rfftfreq(frameSize, 1 / sampleRate) >= result["high_frequency_cutoff_hz"]
	fractions = []
	for start in range(0, len(samples) - frameSize + 1, max(1, int(sampleRate * 0.01))):
		frame = samples[start : start + frameSize]
		if np.sqrt(np.mean(frame * frame)) < result["activity_rms_full_scale"]:
			continue
		power = np.abs(np.fft.rfft((frame - np.mean(frame)) * window)) ** 2
		total = np.sum(power)
		if total > 0:
			fractions.append(float(np.sum(power[highBins]) / total))
	result["active_frames"] = len(fractions)
	if fractions:
		result["mean_high_frequency_energy_fraction"] = float(np.mean(fractions))
		result["median_high_frequency_energy_fraction"] = float(np.median(fractions))
	return result


def _diagnoseCase(voice, piperVoice, text: str, wavPath: Path) -> dict:
	"""Instrument the actual backend path; do not phonemize the input a second time."""
	import numpy as np

	result = {
		"text": text,
		"frontend_seconds": 0.0,
		"id_mapping_seconds": 0.0,
		"inference_seconds": 0.0,
		"time_to_first_audio_seconds": None,
		"frontend_chunks": [],
		"id_mapping_calls": [],
		"audio_chunks": [],
		"raw_inference_audio": [],
		"acoustic_flags": [],
		"missing_symbols": [],
		"failures": [],
	}
	pcmParts = []

	def recordPhonemes(phonemes):
		result["frontend_chunks"].append({"phonemes": list(phonemes), "joined": "".join(phonemes)})

	# Streaming frontends must be timed while advancing the generator, excluding
	# the inference work that happens between its yields.
	if hasattr(piperVoice, "phonemize_stream"):
		frontendName = "phonemize_stream"
		originalFrontend = piperVoice.phonemize_stream

		def timedFrontend(*args, **kwargs):
			started = time.perf_counter()
			try:
				chunks = iter(originalFrontend(*args, **kwargs))
			finally:
				result["frontend_seconds"] += time.perf_counter() - started
			while True:
				started = time.perf_counter()
				try:
					phonemes = next(chunks)
				except StopIteration:
					return
				finally:
					result["frontend_seconds"] += time.perf_counter() - started
				recordPhonemes(phonemes)
				yield phonemes

	else:
		frontendName = "phonemize"
		originalFrontend = piperVoice.phonemize

		def timedFrontend(*args, **kwargs):
			started = time.perf_counter()
			try:
				chunks = originalFrontend(*args, **kwargs)
			finally:
				result["frontend_seconds"] += time.perf_counter() - started
			for phonemes in chunks:
				recordPhonemes(phonemes)
			return chunks

	originalMapping = piperVoice.phonemes_to_ids

	def timedMapping(phonemes):
		missing = sorted(set(phonemes) - piperVoice.config.phoneme_id_map.keys())
		result["missing_symbols"] = sorted(set(result["missing_symbols"]) | set(missing))
		started = time.perf_counter()
		try:
			ids = originalMapping(phonemes)
		finally:
			result["id_mapping_seconds"] += time.perf_counter() - started
		result["id_mapping_calls"].append({"phonemes": list(phonemes), "ids": list(ids), "missing": missing})
		return ids

	originalInference = piperVoice.phoneme_ids_to_audio

	def timedInference(*args, **kwargs):
		started = time.perf_counter()
		try:
			audioResult = originalInference(*args, **kwargs)
		finally:
			result["inference_seconds"] += time.perf_counter() - started
		# Piper normalizes the returned audio later in synthesize; retain its original amplitude.
		rawAudio = np.asarray(
			audioResult[0] if isinstance(audioResult, tuple) else audioResult, dtype=np.float64
		)
		result["raw_inference_audio"].append(
			{
				"sample_count": int(rawAudio.size),
				"peak": float(np.max(np.abs(rawAudio))) if rawAudio.size else 0.0,
				"rms": float(np.sqrt(np.mean(rawAudio * rawAudio))) if rawAudio.size else 0.0,
			}
		)
		return audioResult

	originalSynthesize = piperVoice.synthesize

	def observedSynthesize(*args, **kwargs):
		for chunk in originalSynthesize(*args, **kwargs):
			result["audio_chunks"].append(
				{
					"phonemes": list(chunk.phonemes),
					"joined": "".join(chunk.phonemes),
					"ids": list(chunk.phoneme_ids),
					"sample_count": int(chunk.audio_float_array.size),
				}
			)
			yield chunk

	started = time.perf_counter()
	try:
		with ExitStack() as stack:
			for name, replacement in (
				(frontendName, timedFrontend),
				("phonemes_to_ids", timedMapping),
				("phoneme_ids_to_audio", timedInference),
				("synthesize", observedSynthesize),
			):
				stack.enter_context(patch.object(piperVoice, name, replacement))
			for audio in voice.synthesize(text, voice.defaults, 1.0, lambda: False):
				if result["time_to_first_audio_seconds"] is None:
					result["time_to_first_audio_seconds"] = time.perf_counter() - started
				if audio.sampleRate != voice.sampleRate or audio.channels != 1 or audio.width != 2:
					raise ValueError("Engine did not return the expected mono PCM16 audio")
				if len(audio.data) % 2:
					raise ValueError("Engine returned an incomplete PCM16 sample")
				pcmParts.append(audio.data)
	except Exception:
		result["failures"].append("synthesis_exception")
		result["exception"] = traceback.format_exc()
	result["total_seconds"] = time.perf_counter() - started
	pcm = b"".join(pcmParts)
	samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
	activeThreshold = 0.01
	active = np.flatnonzero(np.abs(samples) >= activeThreshold * 32768)
	result["pcm"] = {
		"sample_rate": voice.sampleRate,
		"sample_count": int(samples.size),
		"duration_seconds": samples.size / voice.sampleRate,
		"peak_pcm16": float(np.max(np.abs(samples))) if samples.size else 0.0,
		"rms_pcm16": float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0,
		"nonzero_samples": int(np.count_nonzero(samples)),
		"activity_threshold_full_scale": activeThreshold,
		"leading_below_threshold_seconds": int(active[0] if active.size else samples.size) / voice.sampleRate,
		"trailing_below_threshold_seconds": (
			int(samples.size - active[-1] - 1 if active.size else samples.size) / voice.sampleRate
		),
	}
	result["acoustics"] = _acousticMetrics(samples / 32768, voice.sampleRate)
	medianHighEnergy = result["acoustics"]["median_high_frequency_energy_fraction"]
	if medianHighEnergy is not None and medianHighEnergy > 0.9:
		result["acoustic_flags"].append("high_frequency_dominant")
	if not samples.size:
		result["failures"].append("empty_audio")
	elif not result["pcm"]["nonzero_samples"]:
		result["failures"].append("silent_audio")
	if result["missing_symbols"]:
		result["failures"].append("missing_phoneme_symbols")
	with wave.open(str(wavPath), "wb") as wav:
		wav.setnchannels(1)
		wav.setsampwidth(2)
		wav.setframerate(voice.sampleRate)
		wav.writeframes(pcm)
	result["wav"] = wavPath.name
	return result


def _runtimeMetadata() -> dict:
	versions = {}
	for name in ("piper-tts", "numpy", "onnxruntime", "transformers", "optimum", "pandas", "pyarrow"):
		try:
			versions[name] = metadata.version(name)
		except metadata.PackageNotFoundError:
			versions[name] = None
	modulePaths = {
		name: getattr(module, "__file__", None)
		for name, module in list(sys.modules.items())
		if name in ("piper", "piper.voice", "piper.phonemize_espeak")
		or name.startswith("piper.enhance_phonemizer")
	}
	return {
		"python": sys.version,
		"executable": sys.executable,
		"platform": platform.platform(),
		"packages": versions,
		"module_paths": modulePaths,
	}


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--short-speech-repeat",
		action="store_true",
		help="Enable Piper's experimental short speech option; requires a prepared model and updated wheel",
	)
	parser.add_argument("--voice", required=True, help="Local .onnx or .onnx.json voice")
	parser.add_argument("--ezafe-model", required=True, help="Local Ezafe model directory")
	parser.add_argument("--homograph-dictionary", required=True, help="Local homograph parquet file")
	parser.add_argument("--output", required=True, type=Path, help="Directory for report.json and WAV files")
	parser.add_argument("--text", action="append", help="Replace default cases; repeat for several inputs")
	parser.add_argument(
		"--hash-assets", action="store_true", help="Include SHA256 of the voice and its config"
	)
	args = parser.parse_args(argv)
	if args.text and any(not text.strip() for text in args.text):
		parser.error("--text must contain non-whitespace text")
	modelPath, configPath = localVoicePaths(args.voice)
	args.output.mkdir(parents=True, exist_ok=True)
	cases = (
		[(f"custom_{index}", text) for index, text in enumerate(args.text, 1)]
		if args.text
		else _defaultCases()
	)
	report = {
		"note": (
			"Audio metrics detect empty/silent output, not correct pronunciation. Listen to the WAV files. "
			"Acoustic flags are heuristic suspicions, not pronunciation failures or proof of correctness. "
			"high_frequency_dominant means median active-frame energy above 4 kHz exceeds 90%. "
			"Raw inference peak/RMS are measured before Piper's audio normalization. "
			"Explicit IPA cases are candidates for comparison, not golden pronunciations. "
			"Both modes share one loaded voice; only use_persian_phonemizer changes between modes. "
			"Times include instrumentation; the first case may include lazy initialization. "
			"This tests the Piper service backend, not NVDA typing cancellation or audio playback."
		),
		"assets": {
			"voice": str(modelPath),
			"config": str(configPath),
			"ezafe_model": str(Path(args.ezafe_model).resolve()),
			"homograph_dictionary": str(Path(args.homograph_dictionary).resolve()),
		},
		"short_speech_repeat": args.short_speech_repeat,
		"modes": {},
	}
	if args.hash_assets:
		for name, path in (("voice_sha256", modelPath), ("config_sha256", configPath)):
			with path.open("rb") as stream:
				report["assets"][name] = hashlib.file_digest(stream, "sha256").hexdigest()
	logging.basicConfig(level=logging.WARNING)
	warnings = _WarningLog()
	logging.getLogger().addHandler(warnings)
	failed = False
	try:
		print("Loading one Piper voice and the local Persian resources for both modes...", flush=True)
		from piper import PiperVoice

		originalLoad = PiperVoice.load
		loaded = []

		def captureLoad(*loadArgs, **loadKwargs):
			voice = originalLoad(*loadArgs, **loadKwargs)
			loaded.append(voice)
			return voice

		started = time.perf_counter()
		with patch.object(PiperVoice, "load", side_effect=captureLoad):
			voice = loadPiperVoice(
				modelPath,
				configPath,
				persianPhonemizer=True,
				shortSpeechRepeat=args.short_speech_repeat,
				ezafeModel=args.ezafe_model,
				homographDictionary=args.homograph_dictionary,
			)
		report["model_load_seconds"] = time.perf_counter() - started
		report["load_warnings"] = warnings.messages[:]
		piperVoice = loaded[0]
		print(f"Loaded voice and Persian resources in {report['model_load_seconds']:.3f}s", flush=True)
		for mode in ("standard", "enhanced"):
			modeResult = report["modes"][mode] = {"cases": []}
			piperVoice.use_persian_phonemizer = mode == "enhanced"
			modeResult["settings"] = asdict(voice.defaults)
			modeResult["providers"] = piperVoice.session.get_providers()
			warningStart = len(warnings.messages)
			try:
				for caseId, text in cases:
					warningStart = len(warnings.messages)
					result = _diagnoseCase(voice, piperVoice, text, args.output / f"{mode}_{caseId}.wav")
					result["case"] = caseId
					result["warnings"] = warnings.messages[warningStart:]
					modeResult["cases"].append(result)
					failed |= bool(result["failures"])
					print(
						f"{mode}/{caseId}: {result['pcm']['duration_seconds']:.3f}s audio, "
						f"{result['frontend_seconds']:.3f}s frontend, "
						f"{result['inference_seconds']:.3f}s inference, failures={result['failures']}, "
						f"acoustic_flags={result['acoustic_flags']}",
						flush=True,
					)
			except Exception:
				failed = True
				modeResult["exception"] = traceback.format_exc()
				modeResult["warnings"] = warnings.messages[warningStart:]
				logging.exception("Diagnostic mode %s failed", mode)
	except Exception:
		failed = True
		report["exception"] = traceback.format_exc()
		logging.exception("Diagnostic initialization failed")
	finally:
		logging.getLogger().removeHandler(warnings)
		report["runtime"] = _runtimeMetadata()
		report["automated_checks_passed"] = not failed
		report["acoustic_flags_detected"] = any(
			case["acoustic_flags"] for mode in report["modes"].values() for case in mode["cases"]
		)
		reportPath = args.output / "report.json"
		reportPath.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	print(f"Report: {reportPath.resolve()}")
	return int(failed)


if __name__ == "__main__":
	raise SystemExit(main())
