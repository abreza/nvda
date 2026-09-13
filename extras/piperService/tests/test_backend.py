# A part of NonVisual Desktop Access (NVDA)
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Keep optional Piper engine behavior at the engine boundary."""

import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from sonata_piper.backend import loadPiperVoice
from sonata_piper.server import main


class PiperOptionTests(unittest.TestCase):
	def setUp(self):
		directory = tempfile.TemporaryDirectory()
		self.addCleanup(directory.cleanup)
		self.model = Path(directory.name) / "mana.onnx"
		self.configPath = Path(f"{self.model}.json")
		self.configPath.write_text("{}", encoding="utf-8")
		self.pcm = b"\x01\x00\x02\x00"
		self.engine = SimpleNamespace(
			config=SimpleNamespace(
				espeak_voice="fa",
				sample_rate=22050,
				num_speakers=1,
				speaker_id_map={},
				length_scale=1.0,
				noise_scale=0.667,
				noise_w_scale=0.8,
			),
			synthesize=mock.Mock(
				return_value=iter(
					[
						SimpleNamespace(
							audio_int16_bytes=self.pcm,
							sample_rate=22050,
							sample_channels=1,
							sample_width=2,
						),
					],
				),
			),
		)
		self.loader = mock.Mock(return_value=self.engine)
		piper = ModuleType("piper")
		piper.PiperVoice = SimpleNamespace(load=self.loader)
		piperConfig = ModuleType("piper.config")
		piperConfig.SynthesisConfig = lambda **values: SimpleNamespace(**values)
		modules = mock.patch.dict("sys.modules", {"piper": piper, "piper.config": piperConfig})
		modules.start()
		self.addCleanup(modules.stop)

	def test_enabled_default_is_forwarded_to_piper(self):
		loadPiperVoice(self.model, self.configPath)
		self.assertTrue(self.loader.call_args.kwargs["use_short_speech_repeat"])

	def test_disabled_option_does_not_send_new_keyword_to_older_wheels(self):
		loadPiperVoice(self.model, self.configPath, shortSpeechRepeat=False)
		self.assertNotIn("use_short_speech_repeat", self.loader.call_args.kwargs)

	def test_enabled_option_is_forwarded_without_service_synthesis_changes(self):
		voice = loadPiperVoice(self.model, self.configPath, shortSpeechRepeat=True)
		self.assertTrue(self.loader.call_args.kwargs["use_short_speech_repeat"])
		cancelled = mock.Mock(return_value=False)
		[audio] = voice.synthesize("دستیار", voice.defaults, 0.5, cancelled)
		self.assertEqual(audio.data, self.pcm)
		self.assertEqual(self.engine.synthesize.call_args.args[0], "دستیار")
		self.assertEqual(self.engine.synthesize.call_args.args[1].volume, 0.5)
		self.assertIs(self.engine.synthesize.call_args.kwargs["cancelled_callback"], cancelled)

	def test_cli_enables_repetition_by_default_and_accepts_both_switches(self):
		voice = loadPiperVoice(self.model, self.configPath)
		for extra, enabled in (
			([], True),
			(["--short-speech-repeat"], True),
			(["--no-short-speech-repeat"], False),
		):
			with (
				self.subTest(enabled=enabled),
				mock.patch("sonata_piper.server.configuredVoices", return_value={"default": voice}) as loader,
				mock.patch("sonata_piper.server.createServer"),
				mock.patch("sonata_piper.server.logging.basicConfig"),
			):
				main(["--voice", "mana.onnx", *extra])
				self.assertEqual(loader.call_args.kwargs["shortSpeechRepeat"], enabled)


if __name__ == "__main__":
	unittest.main()
