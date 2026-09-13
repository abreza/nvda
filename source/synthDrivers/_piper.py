# A part of NonVisual Desktop Access (NVDA)
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Queued Piper synthesis and interruptible NVDA audio playback."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from queue import Empty, Queue
from threading import Condition, Event, Thread
from typing import TYPE_CHECKING

import nvwave
from logHandler import log
from synthDriverHandler import synthDoneSpeaking, synthIndexReached

if TYPE_CHECKING:
	from piper import PiperVoice
	from synthDriverHandler import SynthDriver


def isPiperAvailable() -> bool:
	"""Check for Piper without loading its inference engine during driver discovery."""
	return find_spec("piper") is not None


def loadVoice(
	modelPath: Path,
	configPath: Path,
	usePersianPhonemizer: bool,
	ezafeModelPath: str | None,
) -> PiperVoice:
	"""Load a voice from the custom Piper wheel, propagating initialization failures."""
	from piper import PiperVoice

	return PiperVoice.load(
		model_path=str(modelPath),
		config_path=str(configPath),
		use_cuda=False,
		use_persian_phonemizer=usePersianPhonemizer,
		ezafe_model_path=ezafeModelPath,
	)


@dataclass(frozen=True)
class Speech:
	"""A text segment and the voice/settings in effect when it was submitted."""

	text: str
	voice: PiperVoice
	speakerId: int | None
	lengthScale: float
	volume: float


@dataclass(frozen=True)
class Index:
	"""An NVDA speech index following all previously submitted audio."""

	index: int


@dataclass(frozen=True)
class Silence:
	"""A timed break, expressed in milliseconds."""

	duration: int
	sampleRate: int


type SpeechItem = Speech | Index | Silence


class SpeechWorker(Thread):
	"""Own the synthesis queue and player; allow the caller to interrupt playback."""

	def __init__(self, synth: SynthDriver, outputDevice: str) -> None:
		super().__init__(name="Piper", daemon=True)
		self._synth = synth
		self._outputDevice = outputDevice
		self._queue: Queue[tuple[Event, tuple[SpeechItem, ...]] | None] = Queue()
		self._state = Condition()
		self._cancelled = Event()
		self._isStopping = False
		self._isPaused = False
		self._cancelling = 0
		self._player: nvwave.WavePlayer | None = None
		self._format: tuple[int, int, int] | None = None

	def speak(self, items: list[SpeechItem]) -> None:
		"""Submit an entire utterance atomically, including its index boundaries."""
		with self._state:
			if not self._isStopping:
				self._queue.put((self._cancelled, tuple(items)))

	def cancel(self) -> None:
		"""Invalidate active/pending speech and interrupt any blocking audio operation."""
		with self._state:
			self._cancelling += 1
			self._cancelled.set()
			self._cancelled = Event()
			self._isPaused = False
			while True:
				try:
					if self._queue.get_nowait() is None:
						self._queue.put(None)
						break
				except Empty:
					break
			player = self._player
			self._state.notify_all()
		# feed/idle must never hold _state: stop releases their native waits.
		try:
			if player is not None:
				player.stop()
		finally:
			with self._state:
				self._cancelling -= 1
				self._state.notify_all()

	def pause(self, switch: bool) -> None:
		"""Pause or resume playback and queued synthesis without discarding speech."""
		with self._state:
			self._isPaused = switch
			if self._player is not None:
				self._player.pause(switch)
			self._state.notify_all()

	def stop(self) -> None:
		"""Cancel playback and request worker shutdown."""
		with self._state:
			if self._isStopping:
				return
			self._isStopping = True
		try:
			self.cancel()
		finally:
			self._queue.put(None)

	def _waitUntilReady(self, cancelled: Event) -> bool:
		with self._state:
			self._state.wait_for(
				lambda: (
					cancelled.is_set() or self._isStopping or (not self._isPaused and not self._cancelling)
				),
			)
			return not cancelled.is_set() and not self._isStopping

	def run(self) -> None:
		"""Process utterances in order and release audio resources before exiting."""
		try:
			while (utterance := self._queue.get()) is not None:
				cancelled, items = utterance
				if not self._waitUntilReady(cancelled):
					continue
				try:
					self._speak(items, cancelled)
				except Exception:
					log.exception("Error synthesizing Piper speech")
					if self._player is not None:
						self._player.stop()
				# Completion is a queue boundary, never a text-segment boundary.
				with self._state:
					isDone = not cancelled.is_set() and not self._isStopping and self._queue.empty()
					if isDone:
						synthDoneSpeaking.notify(synth=self._synth)
		finally:
			with self._state:
				player, self._player = self._player, None
			if player is not None:
				player.close()

	def _speak(self, items: tuple[SpeechItem, ...], cancelled: Event) -> None:
		for item in items:
			if not self._waitUntilReady(cancelled):
				return
			if isinstance(item, Speech):
				self._synthesize(item, cancelled)
			elif isinstance(item, Silence):
				self._silence(item, cancelled)
			else:
				if self._player is not None:
					self._player.sync()
				if self._waitUntilReady(cancelled):
					with self._state:
						if not cancelled.is_set() and not self._isStopping:
							synthIndexReached.notify(synth=self._synth, index=item.index)
		if not cancelled.is_set() and self._player is not None:
			self._player.idle()

	def _synthesize(self, speech: Speech, cancelled: Event) -> None:
		from piper.config import SynthesisConfig

		settings = SynthesisConfig(
			speaker_id=speech.speakerId,
			length_scale=speech.lengthScale,
			volume=speech.volume,
		)
		for chunk in speech.voice.synthesize(speech.text, settings, cancelled_callback=cancelled.is_set):
			if not self._waitUntilReady(cancelled):
				return
			self._feed(
				chunk.audio_int16_bytes,
				(chunk.sample_channels, chunk.sample_rate, 16),
				cancelled,
			)

	def _silence(self, silence: Silence, cancelled: Event) -> None:
		# Limit buffer size so long breaks remain interruptible.
		remaining = silence.sampleRate * silence.duration // 1000
		while remaining > 0 and self._waitUntilReady(cancelled):
			samples = min(remaining, max(1, silence.sampleRate // 20))
			self._feed(bytes(samples * 2), (1, silence.sampleRate, 16), cancelled, isSilence=True)
			remaining -= samples

	def _feed(
		self,
		data: bytes,
		audioFormat: tuple[int, int, int],
		cancelled: Event,
		*,
		isSilence: bool = False,
	) -> None:
		if not self._waitUntilReady(cancelled):
			return
		if self._player is not None and self._format != audioFormat:
			self._player.idle()
			with self._state:
				oldPlayer, self._player = self._player, None
			oldPlayer.close()
		with self._state:
			if cancelled.is_set() or self._isStopping:
				return
			if self._player is None:
				channels, sampleRate, bitsPerSample = audioFormat
				self._player = nvwave.WavePlayer(
					channels=channels,
					samplesPerSec=sampleRate,
					bitsPerSample=bitsPerSample,
					outputDevice=self._outputDevice,
				)
				self._format = audioFormat
				self._player.pause(self._isPaused)
			player = self._player
		if not self._waitUntilReady(cancelled):
			return
		if isSilence:
			player.startTrimmingLeadingSilence(False)
		player.feed(data)
		# Discard audio if cancellation raced with entry into the native feed call.
		if cancelled.is_set():
			player.stop()
