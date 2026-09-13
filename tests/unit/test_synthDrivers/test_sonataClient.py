# A part of NonVisual Desktop Access (NVDA)
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Real loopback RPC regression tests, also runnable without NVDA's native runtime."""

import unittest
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread

import grpc
from synthDrivers._sonata import sonata_grpc_pb2 as messages
from synthDrivers._sonata.client import Client, validateEndpoint


class TestSonataClient(unittest.TestCase):
	def setUp(self):
		self.loads = 0
		self.remoteId = "handle-1"
		self.options = []
		self.requests = []
		self.audio = [b"\x01\x00" * 20]
		self.entered = Event()
		self.released = Event()
		self.cancelledOnServer = Event()
		self.block = False
		self.blockMethod = None
		self.failAfterAudio = False
		self.unavailableOptions = 0
		self.sampleRate = 22050
		self.sampleWidth = 2
		self.pool = ThreadPoolExecutor(max_workers=4)
		self.port = 0
		self._startServer()
		self.client = Client(f"127.0.0.1:{self.port}", timeout=2)
		self.threads = []

	def _startServer(self):
		self.server = grpc.server(self.pool)
		self.server.add_generic_rpc_handlers(
			(
				grpc.method_handlers_generic_handler(
					"sonata_grpc.sonata_grpc",
					{
						"LoadVoice": grpc.unary_unary_rpc_method_handler(
							self._load,
							request_deserializer=messages.VoicePath.FromString,
							response_serializer=messages.VoiceInfo.SerializeToString,
						),
						"SetSynthesisOptions": grpc.unary_unary_rpc_method_handler(
							self._setOptions,
							request_deserializer=messages.VoiceSynthesisOptions.FromString,
							response_serializer=messages.SynthesisOptions.SerializeToString,
						),
						"SynthesizeUtterance": grpc.unary_stream_rpc_method_handler(
							self._speak,
							request_deserializer=messages.Utterance.FromString,
							response_serializer=messages.SynthesisResult.SerializeToString,
						),
					},
				),
			),
		)
		self.port = self.server.add_insecure_port(f"127.0.0.1:{self.port}")
		self.server.start()

	def tearDown(self):
		self.released.set()
		self.client.close()
		self.server.stop(0).wait(3)
		for thread in self.threads:
			thread.join(3)
			self.assertFalse(thread.is_alive(), "Client failed to stop")
		self.pool.shutdown(wait=True)

	def _load(self, request, context):
		self.loads += 1
		self._waitIfBlocked("load", context)
		return messages.VoiceInfo(
			voice_id=self.remoteId,
			audio=messages.AudioInfo(
				sample_rate=self.sampleRate,
				num_channels=1,
				sample_width=self.sampleWidth,
			),
			language="fa_IR",
			speakers={0: "Mana", 2: "Other speaker"},
			synth_options=messages.SynthesisOptions(length_scale=1.0),
		)

	def _setOptions(self, request, context):
		self._waitIfBlocked("options", context)
		if self.unavailableOptions:
			self.unavailableOptions -= 1
			context.abort(grpc.StatusCode.UNAVAILABLE, "Service temporarily unavailable")
		if request.voice_id != self.remoteId:
			context.abort(grpc.StatusCode.NOT_FOUND, "Expired voice handle")
		self.options.append(request.synthesis_options)
		return request.synthesis_options

	def _waitIfBlocked(self, method, context):
		if self.blockMethod == method:
			context.add_callback(self.cancelledOnServer.set)
			self.entered.set()
			self.released.wait(3)

	def _speak(self, request, context):
		self.requests.append(request)
		if self.block:
			context.add_callback(self.cancelledOnServer.set)
			self.entered.set()
			self.released.wait(3)
		for chunk in self.audio:
			if not context.is_active():
				return
			yield messages.SynthesisResult(wav_samples=chunk)
		if self.failAfterAudio:
			context.abort(grpc.StatusCode.UNAVAILABLE, "Service stopped after partial audio")

	def _synthesize(self, voice=None, cancelled=None, rate=50, speaker="0"):
		if voice is None:
			voice = self.client.loadVoices(["default"])[0]
		return list(self.client.synthesize(voice, "سلام", speaker, rate, 75, cancelled or Event()))

	def test_wireUsesSpeakerNamesAndSonataRateScale(self):
		voice = self.client.loadVoices(["default"])[0]
		self.assertEqual("default", voice.id)
		self.assertEqual("fa_IR", voice.language)
		for rate, expectedWireRate in ((0, 0), (50, 10), (100, 30)):
			with self.subTest(rate=rate):
				self.assertEqual(self.audio, self._synthesize(voice, rate=rate, speaker="2"))
				self.assertEqual("Other speaker", self.options[-1].speaker)
				self.assertEqual(expectedWireRate, self.requests[-1].speech_args.rate)
				self.assertEqual(75, self.requests[-1].speech_args.volume)

	def test_largeUpstreamAudioIsSplitWithoutChangingSamples(self):
		self.audio = [b"\x02\x00" * 100000]
		chunks = self._synthesize()
		self.assertTrue(all(len(chunk) <= 65536 for chunk in chunks))
		self.assertEqual(self.audio[0], b"".join(chunks))

	def test_reloadsStaleVoiceHandleBeforeAudio(self):
		voice = self.client.loadVoices(["default"])[0]
		self.remoteId = "handle-after-server-restart"
		self.assertEqual(self.audio, self._synthesize(voice))
		self.assertEqual(2, self.loads)
		self.assertEqual(1, len(self.requests))
		self.assertEqual(self.remoteId, self.requests[-1].voice_id)

	def test_reconnectRejectsChangedAudioFormat(self):
		voice = self.client.loadVoices(["default"])[0]
		self.remoteId = "replacement"
		self.sampleRate = 48000
		with self.assertRaisesRegex(ValueError, "voice changed"):
			self._synthesize(voice)
		self.assertFalse(self.requests)

	def _synthesizeInThread(self, voice, cancelled=None):
		output, errors = [], []
		finished = Event()

		def run():
			try:
				output.extend(self._synthesize(voice, cancelled))
			except Exception as error:  # noqa: BLE001 - propagate worker failures to the test thread.
				errors.append(error)
			finally:
				finished.set()

		thread = Thread(target=run, daemon=True)
		self.threads.append(thread)
		thread.start()
		return output, errors, finished

	def _observeDisconnection(self):
		disconnected = Event()

		def onConnectivityChanged(state):
			if state == grpc.ChannelConnectivity.TRANSIENT_FAILURE:
				disconnected.set()

		self.client._channel.subscribe(onConnectivityChanged, try_to_connect=True)
		self.addCleanup(self.client._channel.unsubscribe, onConnectivityChanged)
		return disconnected

	def test_temporaryOutageReloadsVoiceAndRecovers(self):
		voice = self.client.loadVoices(["default"])[0]
		self.server.stop(0).wait(1)
		disconnected = self._observeDisconnection()
		output, errors, finished = self._synthesizeInThread(voice)
		self.assertTrue(disconnected.wait(1), "Client did not observe the stopped service")
		self.remoteId = "handle-after-service-restart"
		self._startServer()
		self.assertTrue(finished.wait(2), "Client did not recover after the service restarted")
		self.assertEqual([], errors)
		self.assertEqual(self.audio, output)
		self.assertEqual(2, self.loads)
		self.assertEqual(1, len(self.requests))
		self.assertEqual(self.remoteId, self.requests[-1].voice_id)

	def test_cancellationInterruptsRecoveryWithoutService(self):
		voice = self.client.loadVoices(["default"])[0]
		self.server.stop(0).wait(1)
		disconnected = self._observeDisconnection()
		cancelled = Event()
		output, errors, finished = self._synthesizeInThread(voice, cancelled)
		self.assertTrue(disconnected.wait(1))
		cancelled.set()
		self.client.cancel()
		self.assertTrue(finished.wait(0.5), "Cancellation waited for service recovery")
		self.assertEqual([], errors)
		self.assertEqual([], output)

	def test_recoveryHonorsTotalRequestDeadline(self):
		voice = self.client.loadVoices(["default"])[0]
		self.client._timeout = 0.15
		self.server.stop(0).wait(1)
		started = time.monotonic()
		with self.assertRaises(grpc.RpcError):
			self._synthesize(voice)
		self.assertLess(time.monotonic() - started, 0.75)
		self.assertFalse(self.requests)

	def test_unavailableAfterPartialAudioIsNotReplayed(self):
		voice = self.client.loadVoices(["default"])[0]
		self.failAfterAudio = True
		stream = self.client.synthesize(voice, "text", "0", 50, 75, Event())
		self.assertEqual(self.audio[0], next(stream))
		with self.assertRaises(grpc.RpcError) as raised:
			next(stream)
		self.assertEqual(grpc.StatusCode.UNAVAILABLE, raised.exception.code())
		self.assertEqual(1, len(self.requests))
		self.assertEqual(1, self.loads)

	def test_recoveryValidatesMetadataEvenWhenHandleIsUnchanged(self):
		voice = self.client.loadVoices(["default"])[0]
		self.unavailableOptions = 1
		self.sampleRate = 48000
		with self.assertRaisesRegex(ValueError, "voice changed"):
			self._synthesize(voice)
		self.assertEqual(2, self.loads)
		self.assertFalse(self.requests)

	def test_cancelEventInterruptsOptionsWait(self):
		voice = self.client.loadVoices(["default"])[0]
		self.blockMethod = "options"
		cancelled = Event()
		output, errors, finished = self._synthesizeInThread(voice, cancelled)
		self.assertTrue(self.entered.wait(1))
		cancelled.set()
		self.assertTrue(finished.wait(0.5), "Cancellation waited for synthesis options")
		self.assertEqual([], errors)
		self.assertEqual([], output)
		self.assertTrue(self.cancelledOnServer.wait(1))

	def test_cancelEventInterruptsStaleVoiceReload(self):
		voice = self.client.loadVoices(["default"])[0]
		self.remoteId = "stale"
		self.blockMethod = "load"
		cancelled = Event()
		output, errors, finished = self._synthesizeInThread(voice, cancelled)
		self.assertTrue(self.entered.wait(1))
		cancelled.set()
		self.assertTrue(finished.wait(0.5), "Cancellation waited for voice reload")
		self.assertEqual([], errors)
		self.assertEqual([], output)
		self.assertTrue(self.cancelledOnServer.wait(1))

	def test_rejectsIncompleteSampleFrame(self):
		self.audio = [b"\x01"]
		with self.assertRaisesRegex(ValueError, "incomplete PCM"):
			self._synthesize()

	def test_rejectsUnsupportedVoiceFormatBeforePlayback(self):
		self.sampleWidth = 4
		with self.assertRaisesRegex(ValueError, "requires PCM16"):
			self.client.loadVoices(["default"])

	def test_cancelInterruptsNetworkWaitAndSuppressesLateAudio(self):
		voice = self.client.loadVoices(["default"])[0]
		self.block = True
		cancelled = Event()
		output, errors = [], []
		finished = Event()

		def run():
			try:
				output.extend(self._synthesize(voice, cancelled))
			except Exception as error:  # noqa: BLE001 - propagate worker failures to the test thread.
				errors.append(error)
			finally:
				finished.set()

		thread = Thread(target=run, daemon=True)
		self.threads.append(thread)
		thread.start()
		self.assertTrue(self.entered.wait(2))
		cancelled.set()
		self.client.cancel()
		self.assertTrue(finished.wait(1), "Cancellation waited for server inference")
		self.assertTrue(self.cancelledOnServer.wait(1))
		self.assertEqual([], output)
		self.assertEqual([], errors)

	def test_deadlineDoesNotWaitForStalledServer(self):
		voice = self.client.loadVoices(["default"])[0]
		self.client._timeout = 0.1
		self.block = True
		with self.assertRaises(grpc.RpcError) as raised:
			self._synthesize(voice)
		self.assertEqual(grpc.StatusCode.DEADLINE_EXCEEDED, raised.exception.code())

	def test_alreadyCancelledSpeechDoesNotContactServer(self):
		voice = self.client.loadVoices(["default"])[0]
		cancelled = Event()
		cancelled.set()
		self.assertEqual([], self._synthesize(voice, cancelled))
		self.assertEqual([], self.options)
		self.assertEqual([], self.requests)

	def test_closingChannelInterruptsWaitWithoutError(self):
		voice = self.client.loadVoices(["default"])[0]
		self.block = True
		output, errors = [], []
		finished = Event()

		def run():
			try:
				output.extend(self._synthesize(voice))
			except Exception as error:  # noqa: BLE001 - propagate worker failures to the test thread.
				errors.append(error)
			finally:
				finished.set()

		thread = Thread(target=run, daemon=True)
		self.threads.append(thread)
		thread.start()
		self.assertTrue(self.entered.wait(2))
		self.client.close()
		self.assertTrue(finished.wait(1))
		self.assertEqual([], output)
		self.assertEqual([], errors)


class TestSonataTransportConfiguration(unittest.TestCase):
	def test_normalizesLoopbackWithoutDNS(self):
		self.assertEqual("127.0.0.1:50051", validateEndpoint("localhost:50051"))
		self.assertEqual("[::1]:50051", validateEndpoint("[::1]:50051"))

	def test_rejectsRemoteOrAmbiguousEndpoints(self):
		for value in (
			"example.com:50051",
			"0.0.0.0:50051",
			"127.0.0.1",
			"user@127.0.0.1:1",
			"localhost:1/path",
		):
			with self.subTest(value=value), self.assertRaises(ValueError):
				validateEndpoint(value)

	def test_standaloneServiceUsesIdenticalProtocolDefinitions(self):
		root = Path(__file__).resolve().parents[3]
		self.assertEqual(
			(root / "source/synthDrivers/_sonata/sonata_grpc_pb2.py").read_bytes(),
			(root / "extras/piperService/sonata_piper/sonata_grpc_pb2.py").read_bytes(),
		)
