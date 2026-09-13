# A part of NonVisual Desktop Access (NVDA)
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Regression tests for Piper queueing, playback control and speech translation."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace
from unittest import mock

from speech.commands import BreakCommand, IndexCommand, RateCommand, VolumeCommand
from synthDrivers import _piper, piper


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
		self.events.append(("stop",))

	def close(self) -> None:
		self.events.append(("close",))

	def pause(self, switch: bool) -> None:
		self.isPaused = switch
		self.events.append(("pause", switch))

	def startTrimmingLeadingSilence(self, start: bool) -> None:
		self.events.append(("trim", start))


def _chunk(data: bytes, sampleRate: int = 22050, channels: int = 1) -> SimpleNamespace:
	return SimpleNamespace(audio_int16_bytes=data, sample_rate=sampleRate, sample_channels=channels)


class TestPiperWorker(unittest.TestCase):
	def setUp(self) -> None:
		self.events: list = []
		self.done = Event()
		self.players: list[_Player] = []
		self.workers: list[_piper.SpeechWorker] = []
		self.releases: list[Event] = []
		self.enterContext(
			mock.patch.dict(
				"sys.modules",
				{"piper.config": SimpleNamespace(SynthesisConfig=SimpleNamespace)},
			),
		)
		self.enterContext(mock.patch.object(_piper.synthIndexReached, "notify", side_effect=self._onIndex))
		self.enterContext(mock.patch.object(_piper.synthDoneSpeaking, "notify", side_effect=self._onDone))
		self.enterContext(mock.patch.object(_piper.nvwave, "WavePlayer", side_effect=self._makePlayer))

	def tearDown(self) -> None:
		for release in self.releases:
			release.set()
		for worker in self.workers:
			worker.stop()
			worker.join(timeout=3)
			self.assertFalse(worker.is_alive(), "Piper worker did not stop")

	def _onIndex(self, *, synth, index: int) -> None:
		self.events.append(("index", index))

	def _onDone(self, *, synth) -> None:
		self.events.append(("done",))
		self.done.set()

	def _makePlayer(self, **kwargs) -> _Player:
		player = _Player(self.events, **kwargs)
		self.players.append(player)
		return player

	def _worker(self) -> _piper.SpeechWorker:
		worker = _piper.SpeechWorker(mock.Mock(), "chosen-output-device")
		self.workers.append(worker)
		worker.start()
		return worker

	@staticmethod
	def _speech(text: str, voice=None) -> _piper.Speech:
		if voice is None:
			voice = SimpleNamespace(synthesize=lambda text, settings, **kwargs: iter([_chunk(text.encode())]))
		return _piper.Speech(text, voice, None, 1.0, 0.8)

	def test_completionFollowsAllAudioAndIndices(self) -> None:
		worker = self._worker()
		worker.speak([self._speech("first"), _piper.Index(7), self._speech("last")])
		self.assertTrue(self.done.wait(3))
		relevant = [event for event in self.events if event[0] in {"audio", "index", "done", "idle"}]
		self.assertEqual(
			[("audio", b"first"), ("index", 7), ("audio", b"last"), ("idle",), ("done",)],
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
		with mock.patch.object(_piper.nvwave, "WavePlayer", return_value=player):
			worker = self._worker()
			worker.speak([self._speech("cancelled"), _piper.Index(99)])
			self.assertTrue(entered.wait(3))

			def cancel() -> None:
				worker.cancel()
				cancelled.set()

			cancelThread = Thread(target=cancel, daemon=True)
			cancelThread.start()
			try:
				self.assertTrue(cancelled.wait(1), "cancel waited behind a blocking audio call")
			finally:
				released.set()
				cancelThread.join(3)
			worker.stop()
			worker.join(3)
		if method == "feed":
			self.assertNotIn(("index", 99), self.events)
		self.assertNotIn(("done",), self.events)

	def test_cancelInterruptsFeed(self) -> None:
		self._checkBlockingCancellation("feed")

	def test_cancelInterruptsIdle(self) -> None:
		self._checkBlockingCancellation("idle")

	def test_cancelDuringInferenceDiscardsLateAudioAndPendingSpeech(self) -> None:
		entered, release = Event(), Event()
		self.releases.append(release)

		def synthesize(text, settings, **kwargs):
			entered.set()
			if not release.wait(3):
				raise TimeoutError("Inference was not released")
			yield _chunk(b"stale")

		worker = self._worker()
		worker.speak([self._speech("old", SimpleNamespace(synthesize=synthesize)), _piper.Index(1)])
		self.assertTrue(entered.wait(3))
		worker.speak([self._speech("pending")])
		worker.cancel()
		worker.speak([self._speech("new"), _piper.Index(2)])
		release.set()
		self.assertTrue(self.done.wait(3))
		self.assertEqual([("audio", b"new")], [event for event in self.events if event[0] == "audio"])
		self.assertNotIn(("index", 1), self.events)
		self.assertIn(("index", 2), self.events)

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

		def synthesize(text, settings, **kwargs):
			yield _chunk(b"first")
			nextChunk.set()
			yield _chunk(b"second")

		player.feed = feed
		with mock.patch.object(_piper.nvwave, "WavePlayer", return_value=player):
			worker = self._worker()
			worker.speak([self._speech("text", SimpleNamespace(synthesize=synthesize)), _piper.Index(5)])
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
		self.assertNotIn(("stop",), self.events)

	def test_pauseBeforePlaybackDoesNotCancelUtterance(self) -> None:
		worker = self._worker()
		worker.pause(True)
		worker.speak([self._speech("retained"), _piper.Index(3)])
		worker.pause(False)
		self.assertTrue(self.done.wait(3))
		self.assertIn(("audio", b"retained"), self.events)
		self.assertIn(("index", 3), self.events)

	def test_stopInterruptsPlaybackAndJoins(self) -> None:
		entered, release = Event(), Event()
		self.releases.append(release)
		player = self._makePlayer()

		def feed(data) -> None:
			entered.set()
			if not release.wait(3):
				raise TimeoutError("Shutdown did not interrupt playback")

		player.feed = feed
		player.stop = release.set
		with mock.patch.object(_piper.nvwave, "WavePlayer", return_value=player):
			worker = self._worker()
			worker.speak([self._speech("old")])
			self.assertTrue(entered.wait(3))
			worker.stop()
			worker.join(3)
			self.assertFalse(worker.is_alive())
		self.assertNotIn(("done",), self.events)
		self.assertIn(("close",), self.events)

	def test_fullAudioFormatControlsPlayerReuse(self) -> None:
		voice = SimpleNamespace(
			synthesize=lambda *args, **kwargs: iter([_chunk(b"mono"), _chunk(b"stereo", channels=2)]),
		)
		self._worker().speak([self._speech("formats", voice)])
		self.assertTrue(self.done.wait(3))
		self.assertEqual([1, 2], [player.format["channels"] for player in self.players])

	def test_breakInsertsRequestedSilence(self) -> None:
		self._worker().speak([_piper.Silence(125, 16000)])
		self.assertTrue(self.done.wait(3))
		audio = b"".join(event[1] for event in self.events if event[0] == "audio")
		self.assertEqual(bytes(4000), audio)
		self.assertIn(("trim", False), self.events)

	def test_emptyUtteranceStillCompletes(self) -> None:
		self._worker().speak([])
		self.assertTrue(self.done.wait(3))
		self.assertEqual([], self.players)

	def test_synthesisFailureDoesNotDiscardFollowingUtterance(self) -> None:
		worker = _piper.SpeechWorker(mock.Mock(), "test-device")
		self.workers.append(worker)
		failedVoice = SimpleNamespace(synthesize=mock.Mock(side_effect=RuntimeError("Inference failed")))
		worker.speak([self._speech("failed", failedVoice)])
		worker.speak([self._speech("next"), _piper.Index(6)])
		worker.start()
		self.assertTrue(self.done.wait(3))
		self.assertIn(("audio", b"next"), self.events)
		self.assertIn(("index", 6), self.events)
		self.assertEqual(1, self.events.count(("done",)))

	def test_newSpeechWaitsForCancellationToFinish(self) -> None:
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
			yield _chunk(b"next")

		self.players[0].stop = stop
		cancelThread = Thread(target=worker.cancel, daemon=True)
		cancelThread.start()
		try:
			self.assertTrue(stopping.wait(3))
			worker.speak([self._speech("next", SimpleNamespace(synthesize=synthesize))])
			self.assertFalse(generated.wait(0.05), "New speech started before the previous stop completed")
		finally:
			release.set()
			cancelThread.join(3)
		self.assertTrue(self.done.wait(3))
		self.assertIn(("audio", b"next"), self.events)


class TestPiperDriver(unittest.TestCase):
	def test_optionalPersianModelsAreNotRequiredForInitialSelection(self) -> None:
		driver = object.__new__(piper.SynthDriver)

		def scan() -> None:
			driver._voiceData = {"voice": mock.Mock()}

		with (
			mock.patch.object(piper.BaseSynthDriver, "__init__", return_value=None),
			mock.patch.object(driver, "_scanVoices", side_effect=scan),
			mock.patch.object(driver, "_selectInitialVoice") as select,
			mock.patch.object(_piper, "SpeechWorker"),
		):
			driver.__init__()
			select.assert_called_once()
			self.assertFalse(driver._usePersianPhonemizer)

	def _driverState(self) -> SimpleNamespace:
		return SimpleNamespace(
			_worker=mock.Mock(),
			_currentVoice=SimpleNamespace(config=SimpleNamespace(length_scale=1.0)),
			_currentVoiceId="voice",
			_voiceData={"voice": SimpleNamespace(numSpeakers=3, sampleRate=22050)},
			_variant="2",
			_rate=50,
			_volume=80,
			_rateToLengthScale=piper.SynthDriver._rateToLengthScale,
		)

	def test_rateAndVolumeCommandsOnlyAffectFollowingText(self) -> None:
		driver = self._driverState()
		with (
			mock.patch.object(RateCommand, "defaultValue", new_callable=mock.PropertyMock, return_value=50),
			mock.patch.object(VolumeCommand, "defaultValue", new_callable=mock.PropertyMock, return_value=80),
		):
			piper.SynthDriver.speak(
				driver,
				["first", RateCommand(offset=50), "fast", VolumeCommand(multiplier=0.5), "quiet"],
			)
		items = driver._worker.speak.call_args.args[0]
		self.assertEqual(["first", "fast", "quiet"], [item.text for item in items])
		self.assertEqual([1.0, 0.5, 0.5], [item.lengthScale for item in items])
		self.assertEqual([0.8, 0.8, 0.4], [item.volume for item in items])

	def test_voiceAndSpeakerAreCapturedForQueuedSpeech(self) -> None:
		driver = self._driverState()
		originalVoice = driver._currentVoice
		piper.SynthDriver.speak(driver, ["text", IndexCommand(8), BreakCommand(100)])
		items = driver._worker.speak.call_args.args[0]
		driver._currentVoice = mock.Mock()
		driver._variant = "0"
		self.assertIs(originalVoice, items[0].voice)
		self.assertEqual(2, items[0].speakerId)
		self.assertEqual(_piper.Index(8), items[1])
		self.assertEqual(_piper.Silence(100, 22050), items[2])

	def test_rateMidpointUsesNormalVoiceSpeed(self) -> None:
		self.assertEqual(
			[2.0, 1.0, 0.5],
			[piper.SynthDriver._rateToLengthScale(rate) for rate in (0, 50, 100)],
		)

	def test_pauseDelegatesToPlayback(self) -> None:
		driver = self._driverState()
		piper.SynthDriver.pause(driver, True)
		piper.SynthDriver.pause(driver, False)
		self.assertEqual([mock.call(True), mock.call(False)], driver._worker.pause.call_args_list)
		driver._worker.cancel.assert_not_called()

	def test_invalidVoiceMetadataIsIgnoredAndLocalePreserved(self) -> None:
		with TemporaryDirectory() as directory:
			path = Path(directory)
			for name in ("valid", "invalid"):
				(path / f"{name}.onnx").touch()
			(path / "valid.onnx.json").write_text(
				json.dumps({"audio": {"sample_rate": 22050}, "language": {"code": "fa_IR"}}),
			)
			(path / "invalid.onnx.json").write_text('{"audio": {"sample_rate": -1}}')
			driver = SimpleNamespace(_voiceDir=path, _voiceData={})
			piper.SynthDriver._scanVoices(driver)
			self.assertEqual(["valid"], list(driver._voiceData))
			self.assertEqual("fa_IR", driver._voiceData["valid"].language)

	def test_missingVoiceDirectoryIsNotCreated(self) -> None:
		with TemporaryDirectory() as directory:
			path = Path(directory) / "missing"
			driver = SimpleNamespace(_voiceDir=path, _voiceData={})
			piper.SynthDriver._scanVoices(driver)
			self.assertFalse(path.exists())

	def test_variantRejectsUnknownSpeaker(self) -> None:
		driver = SimpleNamespace(availableVariants={"0": mock.Mock()}, _variant="0")
		with self.assertRaises(ValueError):
			piper.SynthDriver._set_variant(driver, "9")
		self.assertEqual("0", driver._variant)

	def test_terminateJoinsWorkerAndSavesSettingsBeforeReleasingVoices(self) -> None:
		driver = object.__new__(piper.SynthDriver)
		order = []
		driver._worker = SimpleNamespace(stop=lambda: order.append("stop"), join=lambda: order.append("join"))
		driver._loadedVoices = {("voice", True): object()}
		driver._currentVoice = object()
		with mock.patch.object(
			piper.BaseSynthDriver,
			"terminate",
			create=True,
			side_effect=lambda: order.append("settings"),
		):
			driver.terminate()
		self.assertEqual(["stop", "join", "settings"], order)
		self.assertIsNone(driver._worker)
		self.assertIsNone(driver._currentVoice)
		self.assertEqual({}, driver._loadedVoices)
