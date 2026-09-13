# A part of NonVisual Desktop Access (NVDA)
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Opt-in checks of real Persian synthesis through the loopback Sonata service.

Set SONATA_TEST_VOICE to the local Mana .onnx or .onnx.json file,
SONATA_TEST_EZAFE_MODEL to the local Ezafe model directory, and
SONATA_TEST_HOMOGRAPH_DICTIONARY to the local parquet dictionary before running
``python -m unittest discover -s tests -p test_mana.py -v``.
For the experimental engine path, use a prepared voice and also set
SONATA_TEST_SHORT_SPEECH_REPEAT=1.

These checks locate lost text, silence, or altered PCM at the service boundary.
They do not judge pronunciation or simulate NVDA playback and typing cancellation.
"""

import os
import time
import unittest
from dataclasses import dataclass, field, replace
from unittest import mock

import grpc

from sonata_piper import sonata_grpc_pb2 as messages
from sonata_piper.backend import Audio, loadPiperVoice, localVoicePaths
from sonata_piper.server import MAX_AUDIO_BYTES, PiperService, createServer

_RESOURCE_VARIABLES = (
	"SONATA_TEST_VOICE",
	"SONATA_TEST_EZAFE_MODEL",
	"SONATA_TEST_HOMOGRAPH_DICTIONARY",
)


@dataclass
class _Capture:
	text: str
	chunks: list[Audio] = field(default_factory=list)
	firstPcmSeconds: float | None = None


@unittest.skipUnless(
	all(os.environ.get(name) for name in _RESOURCE_VARIABLES),
	"Real Mana synthesis requires " + ", ".join(_RESOURCE_VARIABLES),
)
class ManaIntegrationTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		# Loading local Persian resources sets offline environment variables.
		environment = mock.patch.dict(os.environ)
		environment.start()
		cls.addClassCleanup(environment.stop)
		modelPath, configPath = localVoicePaths(os.environ["SONATA_TEST_VOICE"])
		cls.voice = loadPiperVoice(
			modelPath,
			configPath,
			persianPhonemizer=True,
			shortSpeechRepeat=os.environ.get("SONATA_TEST_SHORT_SPEECH_REPEAT") == "1",
			ezafeModel=os.environ["SONATA_TEST_EZAFE_MODEL"],
			homographDictionary=os.environ["SONATA_TEST_HOMOGRAPH_DICTIONARY"],
		)
		cls.captures = []

		def synthesize(text, settings, volume, cancelled):
			capture = _Capture(text)
			cls.captures.append(capture)
			started = time.perf_counter()
			for audio in cls.voice.synthesize(text, settings, volume, cancelled):
				if audio.data and capture.firstPcmSeconds is None:
					capture.firstPcmSeconds = time.perf_counter() - started
				capture.chunks.append(audio)
				yield audio

		observedVoice = replace(cls.voice, synthesize=synthesize)
		cls.server = createServer(PiperService({"default": observedVoice}))
		cls.addClassCleanup(lambda: cls.server.stop(0).wait(timeout=5))
		port = cls.server.add_insecure_port("127.0.0.1:0")
		if port <= 0:
			raise RuntimeError("Could not bind the test service to loopback")
		cls.server.start()
		cls.channel = grpc.insecure_channel(f"127.0.0.1:{port}")
		cls.addClassCleanup(cls.channel.close)
		grpc.channel_ready_future(cls.channel).result(timeout=10)
		cls.voiceInfo = cls.channel.unary_unary(
			"/sonata_grpc.sonata_grpc/LoadVoice",
			request_serializer=messages.VoicePath.SerializeToString,
			response_deserializer=messages.VoiceInfo.FromString,
		)(messages.VoicePath(config_path="default"), timeout=10)

	def test_word_and_typed_letters_preserve_text_and_real_pcm(self):
		self.assertTrue(self.voiceInfo.language.startswith("fa"))
		self.assertEqual(
			self.voiceInfo.audio,
			messages.AudioInfo(sample_rate=self.voice.sampleRate, num_channels=1, sample_width=2),
		)
		for label, text in (
			("dastyaar", "دستیار"),
			("dal", "د"),
			("sin", "س"),
			("te", "ت"),
			("ye", "ی"),
			("alef", "ا"),
			("re", "ر"),
		):
			with self.subTest(text=text):
				captureCount = len(self.captures)
				started = time.perf_counter()
				call = self.channel.unary_stream(
					"/sonata_grpc.sonata_grpc/SynthesizeUtterance",
					request_serializer=messages.Utterance.SerializeToString,
					response_deserializer=messages.SynthesisResult.FromString,
				)(
					messages.Utterance(
						voice_id=self.voiceInfo.voice_id,
						text=text,
						synthesis_mode=messages.MODE_LAZY,
						speech_args=messages.SpeechArgs(rate=10, volume=100, pitch=50),
					),
					timeout=120,
				)
				rpcChunks = []
				firstRpcPcmSeconds = None
				try:
					for result in call:
						if result.wav_samples and firstRpcPcmSeconds is None:
							firstRpcPcmSeconds = time.perf_counter() - started
						rpcChunks.append(result.wav_samples)
				finally:
					call.cancel()

				self.assertEqual(len(self.captures), captureCount + 1)
				capture = self.captures[-1]
				self.assertEqual(capture.text, text)
				self.assertTrue(capture.chunks, "The engine returned no audio chunks")
				for chunk in capture.chunks:
					self.assertEqual(
						(chunk.sampleRate, chunk.channels, chunk.width),
						(self.voice.sampleRate, 1, 2),
					)
					self.assertEqual(len(chunk.data) % 2, 0)
				enginePcm = b"".join(chunk.data for chunk in capture.chunks)
				self.assertTrue(enginePcm, "The engine returned empty PCM")
				self.assertTrue(any(enginePcm), "The engine returned entirely silent PCM")
				self.assertTrue(rpcChunks, "The service returned no PCM")
				for chunk in rpcChunks:
					self.assertGreater(len(chunk), 0)
					self.assertLessEqual(len(chunk), MAX_AUDIO_BYTES)
					self.assertEqual(len(chunk) % 2, 0)
				self.assertEqual(b"".join(rpcChunks), enginePcm, "The service changed the engine's PCM")
				self.assertIsNotNone(capture.firstPcmSeconds)
				self.assertIsNotNone(firstRpcPcmSeconds)
				print(
					f"Mana {label}: engine first PCM {capture.firstPcmSeconds:.3f}s; "
					f"RPC first PCM {firstRpcPcmSeconds:.3f}s; "
					f"audio {len(enginePcm) / (2 * self.voice.sampleRate):.3f}s",
					flush=True,
				)


if __name__ == "__main__":
	unittest.main()
