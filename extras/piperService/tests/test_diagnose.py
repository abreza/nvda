# A part of NonVisual Desktop Access (NVDA)
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Numerical diagnostic checks without voice models or neural inference."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

try:
	import numpy as np
except ImportError:
	np = None

from sonata_piper.backend import Audio
from sonata_piper.diagnose import _acousticMetrics, _diagnoseCase


@unittest.skipIf(np is None, "Acoustic diagnostics require NumPy from the Piper environment")
class AcousticMetricsTests(unittest.TestCase):
	def test_low_frequency_tone_has_little_high_frequency_energy(self):
		sampleRate = 22050
		time = np.arange(sampleRate // 2) / sampleRate
		metrics = _acousticMetrics(0.5 * np.sin(2 * np.pi * 500 * time), sampleRate)
		self.assertGreater(metrics["active_frames"], 0)
		self.assertLess(metrics["mean_high_frequency_energy_fraction"], 0.01)
		self.assertLess(metrics["median_high_frequency_energy_fraction"], 0.01)

	def test_high_frequency_tone_and_hiss_are_dominant(self):
		sampleRate = 22050
		time = np.arange(sampleRate // 2) / sampleRate
		noise = np.random.default_rng(12).normal(size=len(time))
		spectrum = np.fft.rfft(noise)
		spectrum[np.fft.rfftfreq(len(noise), 1 / sampleRate) < 6000] = 0
		hiss = np.fft.irfft(spectrum, n=len(noise)) * 0.1
		for name, samples in (("tone", 0.5 * np.sin(2 * np.pi * 8000 * time)), ("hiss", hiss)):
			with self.subTest(signal=name):
				metrics = _acousticMetrics(samples, sampleRate)
				self.assertGreater(metrics["active_frames"], 0)
				self.assertGreater(metrics["mean_high_frequency_energy_fraction"], 0.99)
				self.assertGreater(metrics["median_high_frequency_energy_fraction"], 0.99)

	def test_empty_silent_and_inactive_audio_have_no_spectral_estimate(self):
		for name, samples in (
			("empty", np.array([])),
			("silent", np.zeros(2205)),
			("inactive", np.full(2205, 0.001)),
		):
			with self.subTest(signal=name):
				metrics = _acousticMetrics(samples, 22050)
				self.assertEqual(metrics["active_frames"], 0)
				self.assertIsNone(metrics["mean_high_frequency_energy_fraction"])
				self.assertIsNone(metrics["median_high_frequency_energy_fraction"])

	def test_diagnostic_records_amplitude_before_normalization_and_preserves_tuple(self):
		sampleRate = 22050
		time = np.arange(2205) / sampleRate
		rawSamples = 0.0001 * np.sin(2 * np.pi * 8000 * time)
		for tupleResult in (False, True):
			with self.subTest(tupleResult=tupleResult), tempfile.TemporaryDirectory() as outputDirectory:
				originalOutput = (rawSamples, None) if tupleResult else rawSamples
				piperVoice = SimpleNamespace(
					config=SimpleNamespace(phoneme_id_map={"a": [1]}),
					phonemize=lambda text: [["a"]],
					phonemes_to_ids=lambda phonemes: [1],
					phoneme_ids_to_audio=lambda ids: originalOutput,
				)

				def piperSynthesize(text):
					for phonemes in piperVoice.phonemize(text):
						ids = piperVoice.phonemes_to_ids(phonemes)
						audioResult = piperVoice.phoneme_ids_to_audio(ids)
						self.assertIs(audioResult, originalOutput)
						audio = audioResult[0] if isinstance(audioResult, tuple) else audioResult
						normalized = audio / np.max(np.abs(audio))
						yield SimpleNamespace(
							phonemes=phonemes, phoneme_ids=ids, audio_float_array=normalized
						)

				piperVoice.synthesize = piperSynthesize

				def backendSynthesize(text, settings, volume, cancelled):
					for chunk in piperVoice.synthesize(text):
						pcm = (chunk.audio_float_array * 32767).astype("<i2").tobytes()
						yield Audio(data=pcm, sampleRate=sampleRate)

				voice = SimpleNamespace(synthesize=backendSynthesize, defaults=None, sampleRate=sampleRate)
				result = _diagnoseCase(voice, piperVoice, "a", Path(outputDirectory) / "sample.wav")
				rawMetrics = result["raw_inference_audio"][0]
				self.assertEqual(rawMetrics["sample_count"], rawSamples.size)
				self.assertAlmostEqual(rawMetrics["peak"], float(np.max(np.abs(rawSamples))))
				self.assertAlmostEqual(rawMetrics["rms"], float(np.sqrt(np.mean(rawSamples**2))))
				self.assertGreater(result["pcm"]["peak_pcm16"], 30000)
				self.assertIn("high_frequency_dominant", result["acoustic_flags"])
				self.assertEqual(result["failures"], [])


if __name__ == "__main__":
	unittest.main()
