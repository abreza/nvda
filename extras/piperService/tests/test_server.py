# A part of NonVisual Desktop Access (NVDA)
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Exercise the public RPC contract over a real loopback gRPC connection."""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import grpc
from sonata_piper import sonata_grpc_pb2 as messages
from sonata_piper.backend import Audio, Settings, Voice, localVoicePaths
from sonata_piper.server import MAX_AUDIO_BYTES, MAX_TEXT_BYTES, PiperService, configuredVoices, createServer


class ServiceTests(unittest.TestCase):
	def setUp(self):
		self.calls = []
		self.audio = b"\x01\x00\xff\xff" * 40000
		self.started = threading.Event()
		self.release = threading.Event()
		self.finished = threading.Event()
		self.cancelObserved = threading.Event()
		self.failure = None
		self.badAudio = None

		def synthesize(text, settings, volume, cancelled):
			self.calls.append((text, settings, volume))
			if text == "holding":
				self.started.set()
				while not self.release.wait(0.01):
					if cancelled():
						self.cancelObserved.set()
						self.finished.set()
						return
			if self.failure:
				raise self.failure
			yield self.badAudio or Audio(self.audio, 22050)
			self.finished.set()

		self.voice = Voice(
			"test", "fa_IR", 22050, {0: "Mana", 1: "Other"}, Settings(0, 1, 0.667, 0.8), synthesize
		)
		self.service = PiperService({"default": self.voice, "test": self.voice})
		self.server = createServer(self.service)
		port = self.server.add_insecure_port("127.0.0.1:0")
		self.assertGreater(port, 0)
		self.server.start()
		self.channel = grpc.insecure_channel(f"127.0.0.1:{port}")
		grpc.channel_ready_future(self.channel).result(timeout=3)
		self.addCleanup(self.channel.close)
		self.addCleanup(lambda: self.server.stop(0).wait(timeout=3))
		self.addCleanup(self.release.set)
		self.voiceId = self.load().voice_id

	def unary(self, name, request, responseType):
		return self.channel.unary_unary(
			f"/sonata_grpc.sonata_grpc/{name}",
			request_serializer=type(request).SerializeToString,
			response_deserializer=responseType.FromString,
		)(request, timeout=3)

	def load(self, alias="default"):
		return self.unary("LoadVoice", messages.VoicePath(config_path=alias), messages.VoiceInfo)

	def options(self, voiceId=None, **values):
		return self.unary(
			"SetSynthesisOptions",
			messages.VoiceSynthesisOptions(
				voice_id=voiceId or self.voiceId,
				synthesis_options=messages.SynthesisOptions(**values),
			),
			messages.SynthesisOptions,
		)

	def synthesize(self, text="hello", voiceId=None, mode=messages.MODE_LAZY, **speechArgs):
		return self.channel.unary_stream(
			"/sonata_grpc.sonata_grpc/SynthesizeUtterance",
			request_serializer=messages.Utterance.SerializeToString,
			response_deserializer=messages.SynthesisResult.FromString,
		)(
			messages.Utterance(
				voice_id=voiceId or self.voiceId,
				text=text,
				speech_args=messages.SpeechArgs(**speechArgs),
				synthesis_mode=mode,
			),
			timeout=3,
		)

	def assertStatus(self, status, callback):
		with self.assertRaises(grpc.RpcError) as raised:
			callback()
		self.assertEqual(raised.exception.code(), status)
		return raised.exception

	def test_metadata_and_version_match_wire_contract(self):
		version = self.unary("GetSonataVersion", messages.Empty(), messages.Version)
		self.assertTrue(version.version.startswith("sonata-piper-service/"))
		info = self.unary("GetVoiceInfo", messages.VoiceIdentifier(voice_id=self.voiceId), messages.VoiceInfo)
		self.assertEqual(info.language, "fa_IR")
		self.assertEqual(info.audio, messages.AudioInfo(sample_rate=22050, num_channels=1, sample_width=2))
		self.assertEqual(dict(info.speakers), {0: "Mana", 1: "Other"})
		self.assertEqual(info.synth_options.speaker, "Mana")
		self.assertTrue(info.HasField("supports_streaming_output"))
		self.assertFalse(info.supports_streaming_output)

	def test_voice_handles_isolate_options_and_share_model(self):
		second = self.load("test")
		self.assertNotEqual(second.voice_id, self.voiceId)
		result = self.options(speaker="Other", length_scale=2, noise_scale=0.4, noise_w=0.6)
		self.assertEqual(result.speaker, "Other")
		secondOptions = self.unary(
			"GetSynthesisOptions",
			messages.VoiceIdentifier(voice_id=second.voice_id),
			messages.SynthesisOptions,
		)
		self.assertEqual(secondOptions.speaker, "Mana")
		self.assertAlmostEqual(secondOptions.length_scale, 1)
		list(self.synthesize())
		list(self.synthesize(voiceId=second.voice_id))
		self.assertEqual(self.calls[0][1].speaker, 1)
		self.assertEqual(self.calls[0][1].lengthScale, 2)
		self.assertEqual(self.calls[1][1].speaker, 0)
		self.assertEqual(self.calls[1][1].lengthScale, 1)

	def test_option_updates_are_atomic_and_validate_names_and_numbers(self):
		for changes in (
			{"speaker": "1"},
			{"speaker": "missing"},
			{"length_scale": 0},
			{"length_scale": float("nan")},
			{"noise_w": float("inf")},
			{"noise_scale": -0.1},
			{"noise_scale": 10.1},
			{"speaker": "Other", "length_scale": -1},
		):
			with self.subTest(changes=changes):
				self.assertStatus(
					grpc.StatusCode.INVALID_ARGUMENT, lambda changes=changes: self.options(**changes)
				)
		self.assertEqual(self.options().speaker, "Mana")

	def test_audio_is_unmodified_pcm_in_bounded_aligned_messages(self):
		chunks = list(self.synthesize("متن فارسی"))
		self.assertEqual(b"".join(chunk.wav_samples for chunk in chunks), self.audio)
		self.assertGreater(len(chunks), 1)
		self.assertTrue(all(0 < len(chunk.wav_samples) <= MAX_AUDIO_BYTES for chunk in chunks))
		self.assertTrue(all(len(chunk.wav_samples) % 2 == 0 for chunk in chunks))
		self.assertEqual(self.calls[0][0], "متن فارسی")

	def test_upstream_rate_mapping_and_zero_volume_are_preserved(self):
		for rate, expectedLength in ((0, 2), (10, 1), (50, 1 / 3), (100, 1 / 5.5)):
			with self.subTest(rate=rate):
				list(self.synthesize(rate=rate, volume=0, pitch=50))
				self.assertAlmostEqual(self.calls[-1][1].lengthScale, expectedLength)
				self.assertEqual(self.calls[-1][2], 0)
		list(self.synthesize())
		self.assertEqual(self.calls[-1][1].lengthScale, 1)
		self.assertEqual(self.calls[-1][2], 1)

	def test_appended_silence_is_bounded_pcm(self):
		chunks = list(self.synthesize(appended_silence_ms=1600))
		data = b"".join(chunk.wav_samples for chunk in chunks)
		self.assertEqual(data, self.audio + bytes(22050 * 1600 // 1000 * 2))
		self.assertTrue(all(len(chunk.wav_samples) <= MAX_AUDIO_BYTES for chunk in chunks))

	def test_rejects_unsupported_or_out_of_range_speech_args(self):
		for args, status in (
			({"pitch": 49}, grpc.StatusCode.UNIMPLEMENTED),
			({"rate": 101}, grpc.StatusCode.INVALID_ARGUMENT),
			({"volume": 101}, grpc.StatusCode.INVALID_ARGUMENT),
			({"appended_silence_ms": 60001}, grpc.StatusCode.INVALID_ARGUMENT),
			({"mode": messages.MODE_PARALLEL}, grpc.StatusCode.UNIMPLEMENTED),
		):
			with self.subTest(args=args):
				self.assertStatus(status, lambda args=args: list(self.synthesize(**args)))
		self.assertEqual(self.calls, [])

	def test_realtime_rpc_does_not_claim_inference_streaming(self):
		call = self.channel.unary_stream(
			"/sonata_grpc.sonata_grpc/SynthesizeUtteranceRealtime",
			request_serializer=messages.Utterance.SerializeToString,
			response_deserializer=messages.WaveSamples.FromString,
		)(messages.Utterance(voice_id=self.voiceId, text="hello"), timeout=3)
		self.assertStatus(grpc.StatusCode.UNIMPLEMENTED, lambda: list(call))

	def test_request_limits_measure_utf8_bytes_and_skip_blank_text(self):
		self.assertStatus(
			grpc.StatusCode.INVALID_ARGUMENT,
			lambda: list(self.synthesize("ا" * (MAX_TEXT_BYTES // 2 + 1))),
		)
		self.assertEqual(list(self.synthesize(" \n\t")), [])
		self.assertEqual(self.calls, [])

	def test_oversized_transport_message_is_rejected(self):
		self.assertStatus(
			grpc.StatusCode.RESOURCE_EXHAUSTED,
			lambda: list(self.synthesize("x" * (MAX_TEXT_BYTES * 2))),
		)
		self.assertEqual(self.calls, [])

	def test_unconfigured_paths_and_unknown_handles_are_rejected(self):
		for alias in ("../unconfigured.onnx", "", r"C:\private\voice.onnx.json"):
			with self.subTest(alias=alias):
				self.assertStatus(grpc.StatusCode.NOT_FOUND, lambda alias=alias: self.load(alias))
		self.assertStatus(grpc.StatusCode.NOT_FOUND, lambda: list(self.synthesize(voiceId="missing")))

	def test_voice_handle_table_is_bounded(self):
		with mock.patch("sonata_piper.server.MAX_SESSIONS", 1):
			self.assertStatus(grpc.StatusCode.RESOURCE_EXHAUSTED, self.load)

	def test_cancellation_reaches_engine_and_new_request_can_run(self):
		call = self.synthesize("holding")
		self.assertTrue(self.started.wait(2))
		call.cancel()
		self.assertTrue(self.cancelObserved.wait(2))
		self.assertTrue(self.finished.wait(2))
		chunks = list(self.synthesize("next"))
		self.assertEqual(b"".join(chunk.wav_samples for chunk in chunks), self.audio)
		self.assertEqual([item[0] for item in self.calls], ["holding", "next"])

	def test_cancelled_request_waiting_for_model_never_runs_inference(self):
		waiting = threading.Event()
		innerLock = threading.Lock()

		class ObservedLock:
			def acquire(self, timeout):
				if innerLock.locked():
					waiting.set()
				return innerLock.acquire(timeout=timeout)

			def release(self):
				innerLock.release()

		self.service._voiceLocks[id(self.voice)] = ObservedLock()
		first = self.synthesize("holding")
		self.assertTrue(self.started.wait(2))
		second = self.synthesize("cancel me")
		self.assertTrue(waiting.wait(2))
		second.cancel()
		self.release.set()
		list(first)
		list(self.synthesize("next"))
		self.assertEqual([item[0] for item in self.calls], ["holding", "next"])

	def test_options_are_snapshotted_before_waiting_for_model(self):
		first = self.synthesize("holding")
		self.assertTrue(self.started.wait(2))
		self.options(speaker="Other", length_scale=2)
		self.release.set()
		list(first)
		list(self.synthesize("next"))
		self.assertEqual(self.calls[0][1].speaker, 0)
		self.assertEqual(self.calls[0][1].lengthScale, 1)
		self.assertEqual(self.calls[1][1].speaker, 1)

	def test_engine_failure_returns_safe_error_and_releases_model(self):
		self.failure = RuntimeError("private model path and text")
		with self.assertLogs("sonata_piper.server", level="ERROR"):
			error = self.assertStatus(grpc.StatusCode.INTERNAL, lambda: list(self.synthesize()))
		self.assertNotIn("private", error.details())
		self.failure = None
		self.assertTrue(list(self.synthesize()))

	def test_engine_audio_must_match_advertised_format(self):
		for audio in (Audio(b"x", 22050), Audio(b"xx", 44100), Audio(b"xx", 22050, channels=2)):
			with self.subTest(audio=audio), self.assertLogs("sonata_piper.server", level="ERROR"):
				self.badAudio = audio
				self.assertStatus(grpc.StatusCode.INTERNAL, lambda: list(self.synthesize()))


class VoiceConfigurationTests(unittest.TestCase):
	def test_preload_aliases_only_and_duplicate_assets_share_model(self):
		with tempfile.TemporaryDirectory() as directory:
			model = Path(directory) / "mana.onnx"
			config = Path(f"{model}.json")
			model.touch()
			config.write_text("{}", encoding="utf-8")
			voice = Voice("mana", "fa", 22050, {0: "Default"}, Settings(None, 1, 0.667, 0.8), mock.Mock())
			with mock.patch("sonata_piper.server.loadPiperVoice", return_value=voice) as loader:
				voices = configuredVoices([str(model), str(config)])
			loader.assert_called_once_with(model.resolve(), config.resolve())
			self.assertIs(voices["default"], voice)
			self.assertIs(voices["mana"], voice)
			self.assertIs(voices["mana.onnx"], voice)
			self.assertIs(voices["mana.onnx.json"], voice)
			self.assertEqual(len(voices), 6)

	def test_missing_model_pair_fails_before_server_starts(self):
		with tempfile.TemporaryDirectory() as directory:
			model = Path(directory) / "missing.onnx"
			model.touch()
			with self.assertRaises(ValueError):
				localVoicePaths(str(model))


if __name__ == "__main__":
	unittest.main()
