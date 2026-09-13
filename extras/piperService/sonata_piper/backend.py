# A part of NonVisual Desktop Access (NVDA)
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Local Piper loading and synthesis, independent of NVDA and the wire protocol."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
	speaker: int | None
	lengthScale: float
	noiseScale: float
	noiseW: float


@dataclass(frozen=True)
class Audio:
	data: bytes
	sampleRate: int
	channels: int = 1
	width: int = 2


@dataclass(frozen=True)
class Voice:
	name: str
	language: str
	sampleRate: int
	speakers: dict[int, str]
	defaults: Settings
	synthesize: Callable[[str, Settings, float, Callable[[], bool]], Iterable[Audio]]


def localVoicePaths(path: str) -> tuple[Path, Path]:
	"""Resolve a configured model/config pair; never called using network input."""
	voicePath = Path(path).expanduser().resolve(strict=True)
	if voicePath.name.endswith(".onnx.json"):
		modelPath = voicePath.with_suffix("")
		configPath = voicePath
	elif voicePath.suffix == ".onnx":
		modelPath = voicePath
		configPath = Path(f"{voicePath}.json")
	else:
		raise ValueError("Each --voice must be an .onnx model or .onnx.json configuration")
	if not modelPath.is_file() or not configPath.is_file():
		raise ValueError(f"Missing model/configuration pair: {modelPath}")
	return modelPath, configPath


def loadPiperVoice(
	modelPath: Path,
	configPath: Path,
	*,
	persianPhonemizer: bool = False,
	ezafeModel: str | None = None,
	homographDictionary: str | None = None,
) -> Voice:
	"""Load model and optional Persian resources before the service starts listening."""
	if configPath.stat().st_size > 4 * 1024 * 1024:
		raise ValueError("Voice configuration exceeds 4 MiB")
	with configPath.open(encoding="utf-8") as stream:
		metadata = json.load(stream)
	if persianPhonemizer:
		if not ezafeModel or not Path(ezafeModel).is_dir():
			raise ValueError("Enhanced Persian synthesis requires a local --ezafe-model directory")
		if not homographDictionary or not Path(homographDictionary).is_file():
			raise ValueError("Enhanced Persian synthesis requires a local --homograph-dictionary file")
		os.environ["HOMOGRAPH_DICT_PATH"] = str(Path(homographDictionary).resolve())
		# These resources are explicitly local; do not resolve a missing model over the network.
		os.environ["HF_HUB_OFFLINE"] = "1"
		os.environ["TRANSFORMERS_OFFLINE"] = "1"

	from piper import PiperVoice
	from piper.config import SynthesisConfig

	piperVoice = PiperVoice.load(
		model_path=str(modelPath),
		config_path=str(configPath),
		use_cuda=False,
		use_persian_phonemizer=persianPhonemizer,
		ezafe_model_path=ezafeModel,
	)
	config = piperVoice.config
	if not 8000 <= config.sample_rate <= 192000:
		raise ValueError("Voice sample rate must be between 8000 and 192000 Hz")
	if not 1 <= config.num_speakers <= 10000:
		raise ValueError("Invalid speaker count")
	speakers = {speakerId: str(name) for name, speakerId in config.speaker_id_map.items()}
	if any(type(speakerId) is not int or not 0 <= speakerId < config.num_speakers for speakerId in speakers):
		raise ValueError("Invalid speaker IDs")
	for speakerId in range(config.num_speakers):
		speakers.setdefault(speakerId, "Default" if config.num_speakers == 1 else str(speakerId))
	if len(set(speakers.values())) != len(speakers):
		raise ValueError("Speaker names must be unique")
	defaults = Settings(
		speaker=0 if config.num_speakers > 1 else None,
		lengthScale=config.length_scale,
		noiseScale=config.noise_scale,
		noiseW=config.noise_w_scale,
	)
	if (
		any(
			not math.isfinite(value) or not 0 <= value <= 10
			for value in (
				defaults.lengthScale,
				defaults.noiseScale,
				defaults.noiseW,
			)
		)
		or defaults.lengthScale == 0
	):
		raise ValueError("Invalid voice inference defaults")
	if persianPhonemizer and config.espeak_voice.startswith("fa"):
		from piper.enhance_phonemizer.correct_phonemes import _load_homograph_data
		from piper.enhance_phonemizer.persian_phonemizer import PersianPhonemizer

		# The wheel's lazy initialization catches errors and falls back to eSpeak.
		# Explicit initialization makes missing Persian resources fail at startup.
		piperVoice.persian_phonemizer = PersianPhonemizer(model_path=ezafeModel)
		words, _, _ = _load_homograph_data()
		if not words:
			raise ValueError("The Persian homograph dictionary could not be loaded or is empty")

	def synthesize(text: str, settings: Settings, volume: float, cancelled: Callable[[], bool]):
		synthesisConfig = SynthesisConfig(
			speaker_id=settings.speaker,
			length_scale=settings.lengthScale,
			noise_scale=settings.noiseScale,
			noise_w_scale=settings.noiseW,
			volume=volume,
		)
		for chunk in piperVoice.synthesize(text, synthesisConfig, cancelled_callback=cancelled):
			if cancelled():
				return
			yield Audio(chunk.audio_int16_bytes, chunk.sample_rate, chunk.sample_channels, chunk.sample_width)

	languageInfo = metadata.get("language", {})
	language = (
		languageInfo.get("code", config.espeak_voice)
		if isinstance(languageInfo, dict)
		else config.espeak_voice
	)
	return Voice(modelPath.stem, str(language), config.sample_rate, speakers, defaults, synthesize)
