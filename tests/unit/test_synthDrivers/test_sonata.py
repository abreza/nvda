# A part of NonVisual Desktop Access (NVDA)
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Speech service queueing, cancellation, playback and NVDA configuration tests."""

import json
import unittest
from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace
from unittest import mock

from speech.commands import BreakCommand, IndexCommand, RateCommand, VolumeCommand
from synthDrivers import sonata
from synthDrivers._sonata import worker as speechWorker
from synthDrivers._sonata.client import Voice


def _voice(**changes) -> Voice:
	return replace(
		Voice("default", "remote-voice", "Test voice", "fa_IR", 22050, 1, 2, {"0": "Default"}),
		**changes,
	)


class _Player:
	def __init__(self, events: list, **kwargs):
		self.events = events
		self.format = kwargs
		self.isPaused = False

	def feed(self, data: bytes) -> None:
		self.events.append(("audio", data))

	def sync(self) -> None:
		self.events.append(("sync",))

	def idle(self) -> None:
		self.events.append(("idle",))

	def stop(self) -> None:
		self.isPaused = False
		self.events.append(("stop",))

	def close(self) -> None:
		self.events.append(("close",))

	def pause(self, switch: bool) -> None:
		self.isPaused = switch
		self.events.append(("pause", switch))

	def startTrimmingLeadingSilence(self, start: bool) -> None:
		self.events.append(("trim", start))


class TestSonataWorker(unittest.TestCase):
	def setUp(self) -> None:
		self.events: list = []
		self.done = Event()
		self.players: list[_Player] = []
		self.workers: list[speechWorker.SpeechWorker] = []
		self.releases: list[Event] = []
		self.client = mock.Mock()
		self.client.synthesize.side_effect = lambda voice, text, **kwargs: iter([text.encode()])
		self.enterContext(
			mock.patch.object(speechWorker.synthIndexReached, "notify", side_effect=self._onIndex),
		)
		self.enterContext(
			mock.patch.object(speechWorker.synthDoneSpeaking, "notify", side_effect=self._onDone),
		)
		self.enterContext(mock.patch.object(speechWorker.nvwave, "WavePlayer", side_effect=self._makePlayer))

	def tearDown(self) -> None:
		for release in self.releases:
			release.set()
		for worker in self.workers:
			worker.stop()
			worker.join(timeout=3)
			self.assertFalse(worker.is_alive(), "Sonata worker did not stop")

	def _onIndex(self, *, synth, index: int) -> None:
		self.events.append(("index", index))

	def _onDone(self, *, synth) -> None:
		self.events.append(("done",))
		self.done.set()

	def _makePlayer(self, **kwargs) -> _Player:
		player = _Player(self.events, **kwargs)
		self.players.append(player)
		return player

	def _worker(self, *, start: bool = True) -> speechWorker.SpeechWorker:
		worker = speechWorker.SpeechWorker(mock.Mock(), "chosen-output-device", self.client)
		self.workers.append(worker)
		if start:
			worker.start()
		return worker

	@staticmethod
	def _speech(text: str, voice: Voice | None = None) -> speechWorker.Speech:
		return speechWorker.Speech(text, voice or _voice(), "0", 50, 80)

	def test_completionFollowsAllAudioAndIndices(self) -> None:
		worker = self._worker()
		worker.speak([self._speech("first"), speechWorker.Index(7), self._speech("last")])
		self.assertTrue(self.done.wait(3))
		relevant = [event for event in self.events if event[0] in {"audio", "sync", "index", "done", "idle"}]
		self.assertEqual(
			[("audio", b"first"), ("sync",), ("index", 7), ("audio", b"last"), ("idle",), ("done",)],
			relevant,
		)
		self.assertEqual("chosen-output-device", self.players[0].format["outputDevice"])

	def _checkBlockingCancellation(self, method: str) -> None:
		entered, released, cancelled = Event(), Event(), Event()
		self.releases.append(released)

		def blocked(*args) -> None:
			entered.set()
			if not released.wait(3):
				raise TimeoutError("Audio was not interrupted")

		player = self._makePlayer()
		setattr(player, method, blocked)
		player.stop = released.set
		with mock.patch.object(speechWorker.nvwave, "WavePlayer", return_value=player):
			worker = self._worker()
			worker.speak([self._speech("cancelled"), speechWorker.Index(99)])
			self.assertTrue(entered.wait(3))

			def cancel() -> None:
				worker.cancel()
				cancelled.set()

			cancelThread = Thread(target=cancel, daemon=True)
			cancelThread.start()
			try:
				self.assertTrue(cancelled.wait(1), "cancel waited behind a blocking audio operation")
			finally:
				released.set()
				cancelThread.join(3)
			worker.stop()
			worker.join(3)
		if method != "idle":
			self.assertNotIn(("index", 99), self.events)
		self.assertNotIn(("done",), self.events)
		self.client.cancel.assert_called()

	def test_cancelInterruptsFeed(self) -> None:
		self._checkBlockingCancellation("feed")

	def test_cancelInterruptsSync(self) -> None:
		self._checkBlockingCancellation("sync")

	def test_cancelInterruptsIdle(self) -> None:
		self._checkBlockingCancellation("idle")

	def test_cancelInterruptsNetworkAndDiscardsLateAudioAndPendingSpeech(self) -> None:
		entered, released = Event(), Event()
		self.releases.append(released)

		def synthesize(voice, text, **kwargs):
			if text == "old":
				entered.set()
				if not released.wait(3):
					raise TimeoutError("Network request was not cancelled")
				yield b"stale"
			else:
				yield text.encode()

		self.client.synthesize.side_effect = synthesize
		self.client.cancel.side_effect = released.set
		worker = self._worker()
		worker.speak([self._speech("old"), speechWorker.Index(1)])
		self.assertTrue(entered.wait(3))
		worker.speak([self._speech("pending")])
		worker.cancel()
		worker.speak([self._speech("new"), speechWorker.Index(2)])
		self.assertTrue(self.done.wait(3))
		self.assertEqual([("audio", b"new")], [event for event in self.events if event[0] == "audio"])
		self.assertNotIn(("index", 1), self.events)
		self.assertIn(("index", 2), self.events)
		self.assertEqual(["old", "new"], [call.args[1] for call in self.client.synthesize.call_args_list])

	def test_pauseAndResumePreserveRemainingSpeech(self) -> None:
		feeding, release, nextChunk = Event(), Event(), Event()
		self.releases.append(release)
		player = self._makePlayer()

		def feed(data) -> None:
			self.events.append(("audio", data))
			if data == b"first":
				feeding.set()
				if not release.wait(3):
					raise TimeoutError("First chunk was not released")

		def synthesize(*args, **kwargs):
			yield b"first"
			nextChunk.set()
			yield b"second"

		player.feed = feed
		self.client.synthesize.side_effect = synthesize
		with mock.patch.object(speechWorker.nvwave, "WavePlayer", return_value=player):
			worker = self._worker()
			worker.speak([self._speech("text"), speechWorker.Index(5)])
			self.assertTrue(feeding.wait(3))
			worker.pause(True)
			release.set()
			self.assertTrue(nextChunk.wait(3))
			self.assertTrue(player.isPaused)
			self.assertNotIn(("audio", b"second"), self.events)
			worker.pause(False)
			self.assertTrue(self.done.wait(3))
		self.assertIn(("audio", b"second"), self.events)
		self.assertIn(("index", 5), self.events)
		self.client.cancel.assert_not_called()

	def test_pauseBeforePlaybackRetainsUtterance(self) -> None:
		worker = self._worker()
		worker.pause(True)
		worker.speak([self._speech("retained"), speechWorker.Index(3)])
		self.client.synthesize.assert_not_called()
		worker.pause(False)
		self.assertTrue(self.done.wait(3))
		self.assertIn(("audio", b"retained"), self.events)
		self.assertIn(("index", 3), self.events)

	def test_stopInterruptsNetworkAndClosesClient(self) -> None:
		entered, released = Event(), Event()
		self.releases.append(released)

		def synthesize(*args, **kwargs):
			entered.set()
			if not released.wait(3):
				raise TimeoutError("Shutdown did not interrupt the request")
			yield b"stale"

		self.client.synthesize.side_effect = synthesize
		self.client.cancel.side_effect = released.set
		worker = self._worker()
		worker.speak([self._speech("old")])
		self.assertTrue(entered.wait(3))
		worker.stop()
		worker.join(3)
		self.assertFalse(worker.is_alive())
		self.assertNotIn(("done",), self.events)
		self.assertEqual([], self.players)
		self.client.close.assert_called_once()

	def test_fullAudioFormatControlsPlayerReuse(self) -> None:
		self._worker().speak(
			[
				self._speech("mono"),
				self._speech("stereo", _voice(numChannels=2)),
				self._speech("different rate", _voice(numChannels=2, sampleRate=16000)),
			],
		)
		self.assertTrue(self.done.wait(3))
		self.assertEqual([1, 2, 2], [player.format["channels"] for player in self.players])
		self.assertEqual([22050, 22050, 16000], [player.format["samplesPerSec"] for player in self.players])
		self.assertEqual(2, self.events.count(("close",)))

	def test_breakUsesVoiceChannelsAndSampleWidth(self) -> None:
		self._worker().speak([speechWorker.Silence(125, 16000, 2, 2)])
		self.assertTrue(self.done.wait(3))
		audio = [event[1] for event in self.events if event[0] == "audio"]
		self.assertEqual(bytes(8000), b"".join(audio))
		self.assertLessEqual(max(map(len, audio)), 3200)
		self.assertIn(("trim", False), self.events)

	def test_emptyUtteranceStillCompletes(self) -> None:
		self._worker().speak([])
		self.assertTrue(self.done.wait(3))
		self.assertEqual([], self.players)

	def test_notificationAllowsConcurrentPauseAndReentrantCancellation(self) -> None:
		worker = self._worker(start=False)
		paused = Event()
		callbackResults = []
		pauseThreads = []

		def onIndex(*, synth, index) -> None:
			# An extension callback must not keep the playback state locked.
			def pause() -> None:
				worker.pause(False)
				paused.set()

			pauseThread = Thread(target=pause, daemon=True)
			pauseThreads.append(pauseThread)
			pauseThread.start()
			callbackResults.append(paused.wait(1))
			if not callbackResults[-1]:
				return
			# Also exercise reentrant cancel/speak from the notification itself.
			worker.cancel()
			worker.speak([self._speech("replacement")])

		speechWorker.synthIndexReached.notify.side_effect = onIndex
		worker.speak([self._speech("first"), speechWorker.Index(1), self._speech("stale")])
		worker.start()
		try:
			self.assertTrue(self.done.wait(3))
			self.assertEqual([True], callbackResults)
			self.assertEqual(
				[("audio", b"first"), ("audio", b"replacement")],
				[event for event in self.events if event[0] == "audio"],
			)
		finally:
			for thread in pauseThreads:
				thread.join(3)

	def test_synthesisFailureDoesNotDiscardFollowingUtterance(self) -> None:
		worker = self._worker(start=False)
		self.client.synthesize.side_effect = [RuntimeError("Request failed"), iter([b"next"])]
		worker.speak([self._speech("failed")])
		worker.speak([self._speech("next"), speechWorker.Index(6)])
		worker.start()
		self.assertTrue(self.done.wait(3))
		self.assertIn(("audio", b"next"), self.events)
		self.assertIn(("index", 6), self.events)
		self.assertEqual(1, self.events.count(("done",)))

	def test_newSpeechWaitsForOldCancellationToFinish(self) -> None:
		worker = self._worker()
		worker.speak([self._speech("first")])
		self.assertTrue(self.done.wait(3))
		self.done.clear()
		stopping, release, generated = Event(), Event(), Event()
		self.releases.append(release)

		def stop() -> None:
			stopping.set()
			if not release.wait(3):
				raise TimeoutError("Cancellation was not released")

		def synthesize(*args, **kwargs):
			generated.set()
			yield b"next"

		self.players[0].stop = stop
		self.client.synthesize.side_effect = synthesize
		cancelThread = Thread(target=worker.cancel, daemon=True)
		cancelThread.start()
		try:
			self.assertTrue(stopping.wait(3))
			worker.speak([self._speech("next")])
			self.assertFalse(generated.wait(0.05), "New speech started before the previous stop completed")
		finally:
			release.set()
			cancelThread.join(3)
		self.assertTrue(self.done.wait(3))
		self.assertIn(("audio", b"next"), self.events)


class TestSonataDriver(unittest.TestCase):
	def _driverState(self) -> SimpleNamespace:
		return SimpleNamespace(
			_worker=mock.Mock(),
			_currentVoiceId="default",
			_voiceData={"default": _voice(speakers={"2": "Second"})},
			_variant="2",
			_rate=50,
			_volume=80,
		)

	def test_rateAndVolumeCommandsAffectOnlyFollowingText(self) -> None:
		driver = self._driverState()
		with (
			mock.patch.object(RateCommand, "defaultValue", new_callable=mock.PropertyMock, return_value=50),
			mock.patch.object(VolumeCommand, "defaultValue", new_callable=mock.PropertyMock, return_value=80),
		):
			sonata.SynthDriver.speak(
				driver,
				[
					"first",
					RateCommand(offset=50),
					"fast",
					VolumeCommand(multiplier=0.5),
					"quiet",
					RateCommand(),
					VolumeCommand(),
					"default",
				],
			)
		items = driver._worker.speak.call_args.args[0]
		self.assertEqual(["first", "fast", "quiet", "default"], [item.text for item in items])
		self.assertEqual([50, 100, 100, 50], [item.rate for item in items])
		self.assertEqual([80, 80, 40, 80], [item.volume for item in items])
		self.assertEqual((50, 80), (driver._rate, driver._volume))

	def test_voiceSpeakerAndAudioFormatAreCapturedForQueuedSpeech(self) -> None:
		driver = self._driverState()
		voice = driver._voiceData["default"]
		sonata.SynthDriver.speak(driver, ["text", IndexCommand(8), BreakCommand(100)])
		items = driver._worker.speak.call_args.args[0]
		driver._voiceData["default"] = _voice(numChannels=2)
		driver._variant = "0"
		self.assertIs(voice, items[0].voice)
		self.assertEqual("2", items[0].speaker)
		self.assertEqual(speechWorker.Index(8), items[1])
		self.assertEqual(speechWorker.Silence(100, 22050, 1, 2), items[2])

	def test_pauseDelegatesWithoutCancelling(self) -> None:
		driver = self._driverState()
		sonata.SynthDriver.pause(driver, True)
		sonata.SynthDriver.pause(driver, False)
		self.assertEqual([mock.call(True), mock.call(False)], driver._worker.pause.call_args_list)
		driver._worker.cancel.assert_not_called()

	def test_checkDoesNotContactService(self) -> None:
		with (
			mock.patch.object(sonata.globalVars.appArgs, "secure", False),
			mock.patch.object(sonata, "find_spec", return_value=object()),
			mock.patch.object(sonata, "Client") as client,
		):
			self.assertTrue(sonata.SynthDriver.check())
			client.assert_not_called()

	def test_missingDependenciesAndSecureModeHideDriver(self) -> None:
		with mock.patch.object(sonata.globalVars.appArgs, "secure", False):
			for result in (None, ModuleNotFoundError("google")):
				with (
					self.subTest(result=result),
					mock.patch.object(
						sonata,
						"find_spec",
						return_value=result,
						side_effect=result if isinstance(result, Exception) else None,
					),
				):
					self.assertFalse(sonata.SynthDriver.check())
		with (
			mock.patch.object(sonata.globalVars.appArgs, "secure", True),
			mock.patch.object(sonata, "find_spec") as find,
		):
			self.assertFalse(sonata.SynthDriver.check())
			find.assert_not_called()

	def test_secureModeRejectsDirectInitialization(self) -> None:
		with (
			mock.patch.object(sonata.globalVars.appArgs, "secure", True),
			mock.patch.object(sonata, "Client") as client,
		):
			with self.assertRaises(RuntimeError):
				sonata.SynthDriver()
			client.assert_not_called()

	def test_failedMetadataLoadClosesClientBeforeRegistration(self) -> None:
		with (
			mock.patch.object(sonata.globalVars.appArgs, "secure", False),
			mock.patch.object(
				sonata,
				"_getConnectionSettings",
				return_value=("localhost:50051", ["mana"], 30),
			),
			mock.patch.object(sonata, "Client") as client,
			mock.patch.object(sonata.BaseSynthDriver, "__init__", return_value=None) as baseInit,
			mock.patch.object(speechWorker, "SpeechWorker") as worker,
		):
			client.return_value.loadVoices.side_effect = TimeoutError("Service not running")
			with self.assertRaises(TimeoutError):
				sonata.SynthDriver()
			client.return_value.close.assert_called_once()
			baseInit.assert_not_called()
			worker.assert_not_called()

	def test_initializationLoadsMetadataBeforeRegisteringAndStartingWorker(self) -> None:
		order = []
		with (
			mock.patch.object(sonata.globalVars.appArgs, "secure", False),
			mock.patch.object(
				sonata,
				"_getConnectionSettings",
				return_value=("localhost:50051", ["mana"], 30),
			),
			mock.patch.object(sonata, "Client") as client,
			mock.patch.object(
				sonata.BaseSynthDriver,
				"__init__",
				side_effect=lambda: order.append("register"),
			),
			mock.patch.object(speechWorker, "SpeechWorker") as worker,
		):

			def load(*args, **kwargs):
				order.append("metadata")
				return [_voice()]

			client.return_value.loadVoices.side_effect = load
			worker.return_value.start.side_effect = lambda: order.append("start")
			driver = sonata.SynthDriver()
			self.assertEqual(["metadata", "register", "start"], order)
			self.assertEqual("default", driver.voice)
			self.assertEqual("0", driver.variant)
			client.return_value.loadVoices.assert_called_once_with(["mana"], timeout=2.0)
			client.return_value.close.assert_not_called()

	def test_failedWorkerStartClosesClientAndUnregisters(self) -> None:
		with (
			mock.patch.object(sonata.globalVars.appArgs, "secure", False),
			mock.patch.object(
				sonata,
				"_getConnectionSettings",
				return_value=("localhost:50051", ["mana"], 30),
			),
			mock.patch.object(sonata, "Client") as client,
			mock.patch.object(sonata.BaseSynthDriver, "__init__", return_value=None),
			mock.patch.object(sonata.SynthDriver, "_unregisterConfigSaveAction") as unregister,
			mock.patch.object(speechWorker, "SpeechWorker") as worker,
		):
			client.return_value.loadVoices.return_value = [_voice()]
			worker.return_value.start.side_effect = RuntimeError("Unable to create worker")
			with self.assertRaises(RuntimeError):
				sonata.SynthDriver()
			client.return_value.close.assert_called_once()
			unregister.assert_called()

	def test_terminateJoinsWorkerBeforeSavingSettings(self) -> None:
		driver = object.__new__(sonata.SynthDriver)
		order = []
		driver._worker = SimpleNamespace(stop=lambda: order.append("stop"), join=lambda: order.append("join"))
		driver._client = mock.Mock()
		driver._voiceData = {"default": _voice()}
		with mock.patch.object(
			sonata.BaseSynthDriver,
			"terminate",
			side_effect=lambda: order.append("settings"),
		):
			driver.terminate()
		self.assertEqual(["stop", "join", "settings"], order)
		self.assertIsNone(driver._worker)
		self.assertIsNone(driver._client)
		self.assertEqual({}, driver._voiceData)


class TestSonataConnectionSettings(unittest.TestCase):
	def setUp(self) -> None:
		self.enterContext(mock.patch.dict(sonata.os.environ, {}, clear=True))
		self.speechConfig = mock.MagicMock()
		self.speechConfig.isSet.return_value = False
		self.enterContext(mock.patch.object(sonata.config, "conf", {"speech": self.speechConfig}))

	def test_defaultsDoNotCreateConfigurationSection(self) -> None:
		self.assertEqual(("127.0.0.1:50051", ["default"], 30), sonata._getConnectionSettings())
		self.speechConfig.__setitem__.assert_not_called()

	def test_explicitEnvironmentOverridesConnectionAndVoices(self) -> None:
		with mock.patch.dict(
			sonata.os.environ,
			{"NVDA_SONATA_ENDPOINT": "127.0.0.1:1234", "NVDA_SONATA_VOICES": json.dumps(["mana", "english"])},
		):
			self.assertEqual(("127.0.0.1:1234", ["mana", "english"], 30), sonata._getConnectionSettings())

	def test_invalidEnvironmentVoiceListsAreRejected(self) -> None:
		for value in ("bad json", '"mana"', "[]", '[""]', "[42]"):
			with (
				self.subTest(value=value),
				mock.patch.dict(sonata.os.environ, {"NVDA_SONATA_VOICES": value}),
				self.assertRaises(ValueError),
			):
				sonata._getConnectionSettings()

	def test_existingConfigurationIsReadWithItsSchema(self) -> None:
		self.speechConfig.isSet.return_value = True
		section = mock.MagicMock()
		section.__getitem__.side_effect = {
			"endpoint": "127.0.0.1:4321",
			"voicePaths": ["mana"],
			"requestTimeout": 10,
		}.__getitem__
		self.speechConfig.__getitem__.return_value = section
		self.assertEqual(("127.0.0.1:4321", ["mana"], 10), sonata._getConnectionSettings())
		section.spec.update.assert_called_once_with(sonata._CONNECTION_CONFIG_SPEC)

	def test_invalidTimeoutIsRejected(self) -> None:
		self.speechConfig.isSet.return_value = True
		section = mock.MagicMock()
		self.speechConfig.__getitem__.return_value = section
		for timeout in (0, 301, float("nan"), True, "30"):
			with self.subTest(timeout=timeout):
				section.__getitem__.side_effect = {
					"endpoint": "127.0.0.1:4321",
					"voicePaths": ["mana"],
					"requestTimeout": timeout,
				}.__getitem__
				with self.assertRaises(ValueError):
					sonata._getConnectionSettings()
