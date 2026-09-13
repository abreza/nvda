# A part of NonVisual Desktop Access (NVDA)
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Streaming pitch and accelerated speech preserve NVDA's playback boundaries."""

from ctypes import c_short
from dataclasses import replace
import math
import unittest
from threading import Event
from unittest import mock

from synthDrivers import _sonic
from synthDrivers._sonata import worker as speechWorker
from synthDrivers._sonata.client import Voice


def _speech(**changes) -> speechWorker.Speech:
	return replace(
		speechWorker.Speech(
			"text",
			Voice("default", "remote", "Test", "fa_IR", 22050, 1, 2, {"0": "Default"}),
			"0",
			50,
			80,
		),
		**changes,
	)


class _BufferedSonic:
	"""Leave one frame buffered so tests detect a missing or misplaced flush."""

	def __init__(self, sampleRate: int, channels: int):
		self.sampleRate = sampleRate
		self.channels = channels
		self.pitch = self.speed = 1.0
		self.buffer = b""
		self.output = b""
		self.flushed = False
		self.frames: list[int] = []

	def writeShort(self, data, frames: int) -> None:
		self.frames.append(frames)
		self.buffer += bytes(data)
		boundary = max(0, len(self.buffer) - self.channels * 2)
		self.output, self.buffer = self.buffer[:boundary], self.buffer[boundary:]

	def readShort(self) -> bytes:
		output, self.output = self.output, b""
		return output

	def flush(self) -> None:
		self.flushed = True
		self.output, self.buffer = self.buffer, b""


class TestSonataAudio(unittest.TestCase):
	def setUp(self) -> None:
		self.events = []
		self.streams: list[_BufferedSonic] = []
		self.client = mock.Mock()
		self.client.synthesize.side_effect = lambda *args, **kwargs: iter([b"\x01\x00\x02\x00"])
		self.worker = speechWorker.SpeechWorker(mock.Mock(), "device", self.client)
		player = mock.Mock()
		player.feed.side_effect = lambda data: self.events.append(("audio", data))
		player.sync.side_effect = lambda: self.events.append(("sync",))
		player.idle.side_effect = lambda: self.events.append(("idle",))
		self.enterContext(mock.patch.object(speechWorker.nvwave, "WavePlayer", return_value=player))
		self.enterContext(
			mock.patch.object(
				speechWorker.synthIndexReached,
				"notify",
				side_effect=lambda **kwargs: self.events.append(("index", kwargs["index"])),
			),
		)
		self.initialize = self.enterContext(mock.patch.object(_sonic, "initialize"))
		self.factory = self.enterContext(
			mock.patch.object(_sonic, "SonicStream", side_effect=self._makeStream),
		)

	def _makeStream(self, sampleRate: int, channels: int) -> _BufferedSonic:
		stream = _BufferedSonic(sampleRate, channels)
		self.streams.append(stream)
		return stream

	def test_neutralPitchBypassesProcessingAndKeepsServerRate(self) -> None:
		self.worker._speak((_speech(rate=75),), Event())
		self.initialize.assert_not_called()
		self.factory.assert_not_called()
		self.assertEqual(75, self.client.synthesize.call_args.kwargs["rate"])
		self.assertEqual([("audio", b"\x01\x00\x02\x00"), ("idle",)], self.events)

	def test_boostAtNormalRateAlsoBypassesProcessing(self) -> None:
		self.worker._speak((_speech(rateBoost=True),), Event())
		self.initialize.assert_not_called()
		self.factory.assert_not_called()
		self.assertEqual(50, self.client.synthesize.call_args.kwargs["rate"])

	def test_pitchChangesPreserveServerRateAndNormalTempo(self) -> None:
		for pitch, multiplier in ((0, 0.5), (25, math.sqrt(0.5)), (75, math.sqrt(2)), (100, 2)):
			with self.subTest(pitch=pitch):
				self.worker._speak((_speech(rate=80, pitch=pitch),), Event())
				self.assertEqual(80, self.client.synthesize.call_args.kwargs["rate"])
				self.assertAlmostEqual(multiplier, self.streams[-1].pitch)
				self.assertEqual(1.0, self.streams[-1].speed)
		self.initialize.assert_called_once_with()

	def test_boostChangesLocalTempoAndRequestsNormalServerRate(self) -> None:
		for rate, multiplier in ((0, 0.5), (25, math.sqrt(0.5)), (75, math.sqrt(6)), (100, 6)):
			with self.subTest(rate=rate):
				self.worker._speak((_speech(rate=rate, rateBoost=True, pitch=75),), Event())
				self.assertEqual(50, self.client.synthesize.call_args.kwargs["rate"])
				self.assertEqual(80, self.client.synthesize.call_args.kwargs["volume"])
				self.assertAlmostEqual(multiplier, self.streams[-1].speed)
				self.assertAlmostEqual(math.sqrt(2), self.streams[-1].pitch)

	def test_audioStreamsBeforeRequestEndsAndTailPrecedesIndex(self) -> None:
		def synthesize(*args, **kwargs):
			yield b"\x01\x00\x02\x00"
			self.assertEqual([("audio", b"\x01\x00")], self.events)
			yield b"\x03\x00"

		self.client.synthesize.side_effect = synthesize
		self.worker._speak((_speech(pitch=75), speechWorker.Index(7)), Event())
		self.assertEqual(
			[
				("audio", b"\x01\x00"),
				("audio", b"\x02\x00"),
				("audio", b"\x03\x00"),
				("sync",),
				("index", 7),
				("idle",),
			],
			self.events,
		)
		self.assertTrue(self.streams[0].flushed)

	def test_cancellationDiscardsTailAndNextSegmentUsesFreshStream(self) -> None:
		cancelled = Event()

		def synthesize(*args, **kwargs):
			yield b"\x01\x00"
			cancelled.set()
			yield b"\x02\x00"

		self.client.synthesize.side_effect = synthesize
		self.worker._speak((_speech(pitch=75), speechWorker.Index(7)), cancelled)
		self.assertFalse(self.streams[0].flushed)
		self.assertEqual([], self.events)
		self.client.synthesize.side_effect = lambda *args, **kwargs: iter([b"\x03\x00"])
		self.worker._speak((_speech(pitch=75), speechWorker.Index(8)), Event())
		self.assertEqual(2, len(self.streams))
		self.assertEqual([("audio", b"\x03\x00"), ("sync",), ("index", 8), ("idle",)], self.events)

	def test_failureDoesNotFlushPartialSpeech(self) -> None:
		def synthesize(*args, **kwargs):
			yield b"\x01\x00"
			raise RuntimeError("Connection lost")

		self.client.synthesize.side_effect = synthesize
		with self.assertRaisesRegex(RuntimeError, "Connection lost"):
			self.worker._speak((_speech(pitch=75), speechWorker.Index(7)), Event())
		self.assertFalse(self.streams[0].flushed)
		self.assertEqual([], self.events)

	def test_stereoWritesFrameCountAndKeepsChannelsTogether(self) -> None:
		speech = _speech(pitch=75)
		speech = replace(speech, voice=replace(speech.voice, numChannels=2, sampleRate=16000))
		data = b"\x01\x00\x02\x00\x03\x00\x04\x00"
		self.client.synthesize.side_effect = lambda *args, **kwargs: iter([data])
		self.worker._speak((speech,), Event())
		self.factory.assert_called_once_with(16000, 2)
		self.assertEqual([2], self.streams[0].frames)
		self.assertEqual(data, b"".join(event[1] for event in self.events if event[0] == "audio"))

	def test_explicitBreakRetainsDurationAfterAcceleratedSpeech(self) -> None:
		self.worker._speak(
			(_speech(rate=100, rateBoost=True), speechWorker.Silence(100, 22050), speechWorker.Index(9)),
			Event(),
		)
		audio = b"".join(event[1] for event in self.events if event[0] == "audio")
		self.assertEqual(b"\x01\x00\x02\x00" + bytes(4410), audio)
		self.assertEqual(1, len(self.streams))
		self.assertEqual([2], self.streams[0].frames)


class TestSonataNativeAudio(unittest.TestCase):
	def test_bundledSonicChangesSpeedAndPitch(self) -> None:
		try:
			_sonic.initialize()
		except OSError as error:
			self.skipTest(f"Bundled Sonic DLL is not available: {error}")
		sampleRate = 22050
		source = bytes(
			(c_short * sampleRate)(
				*(round(12000 * math.sin(2 * math.pi * 220 * i / sampleRate)) for i in range(sampleRate)),
			),
		)
		for rate, pitch, boost, speed, frequency in (
			(100, 50, True, 6, 220),
			(50, 100, False, 1, 440),
			(50, 0, False, 1, 110),
		):
			with self.subTest(rate=rate, pitch=pitch):
				client = mock.Mock()
				client.synthesize.return_value = (
					source[start : start + 2048] for start in range(0, len(source), 2048)
				)
				worker = speechWorker.SpeechWorker(mock.Mock(), "device", client)
				output: list[bytes] = []
				with mock.patch.object(worker, "_feed", side_effect=lambda data, *args: output.append(data)):
					worker._synthesize(_speech(rate=rate, pitch=pitch, rateBoost=boost), Event())
				pcm = b"".join(output)
				samples = (c_short * (len(pcm) // 2)).from_buffer_copy(pcm)
				self.assertAlmostEqual(sampleRate / speed, len(samples), delta=sampleRate * 0.04)
				# Measure the centre to exclude the pitch detector's startup and tail.
				centre = samples[len(samples) // 4 : 3 * len(samples) // 4]
				crossings = sum(left <= 0 < right for left, right in zip(centre, centre[1:]))
				self.assertAlmostEqual(frequency, crossings * sampleRate / len(centre), delta=20)
