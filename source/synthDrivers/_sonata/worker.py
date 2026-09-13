# A part of NonVisual Desktop Access (NVDA)
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Queued speech service requests and interruptible NVDA audio playback."""

from __future__ import annotations

from ctypes import c_short
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Condition, Event, Lock, RLock, Thread
from typing import TYPE_CHECKING

import nvwave
from logHandler import log
from synthDriverHandler import synthDoneSpeaking, synthIndexReached
from synthDrivers import _sonic

from .client import Client, Voice

if TYPE_CHECKING:
	from synthDriverHandler import SynthDriver


@dataclass(frozen=True)
class Speech:
	"""Text and settings captured when NVDA submitted the utterance."""

	text: str
	voice: Voice
	speaker: str
	rate: int
	volume: int
	pitch: int = 50
	rateBoost: bool = False


@dataclass(frozen=True)
class Index:
	"""An NVDA speech index following all previously submitted audio."""

	index: int


@dataclass(frozen=True)
class Silence:
	"""A timed break in the selected voice's PCM format."""

	duration: int
	sampleRate: int
	numChannels: int = 1
	sampleWidth: int = 2


type SpeechItem = Speech | Index | Silence


class SpeechWorker(Thread):
	"""Own the speech queue, remote requests and audio device on one worker thread."""

	def __init__(self, synth: SynthDriver, outputDevice: str, client: Client) -> None:
		super().__init__(name="Sonata", daemon=True)
		self._synth = synth
		self._outputDevice = outputDevice
		self._client = client
		self._queue: Queue[tuple[Event, tuple[SpeechItem, ...]] | None] = Queue()
		self._state = Condition()
		# Notifications may synchronously reenter speak/cancel/pause. Keep them
		# outside _state, while ordering notification delivery against cancellation
		# and new utterances. Lock order is _notifications then _state; neither
		# _state nor _playerControl may be held when entering _notifications.
		self._notifications = RLock()
		# Serialize player construction, pause, stop and close, never feed/idle/sync.
		self._playerControl = Lock()
		self._cancelled = Event()
		self._isStopping = False
		self._isPaused = False
		self._cancelling = 0
		self._player: nvwave.WavePlayer | None = None
		self._format: tuple[int, int, int] | None = None
		self._sonicInitialized = False

	def speak(self, items: list[SpeechItem]) -> None:
		"""Submit one utterance atomically, including its index boundaries."""
		with self._notifications, self._state:
			if not self._isStopping:
				self._queue.put((self._cancelled, tuple(items)))

	def cancel(self) -> None:
		"""Invalidate pending speech and release both network and native audio waits."""
		with self._notifications, self._state:
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
			self._state.notify_all()
		# No state lock may be held here: cancel/stop release the worker's waits.
		# New speech waits for every in-flight cancel to finish, preventing an old
		# cancellation from stopping a subsequent request or its audio.
		try:
			try:
				self._client.cancel()
			finally:
				with self._playerControl:
					if self._player is not None:
						self._player.stop()
		finally:
			with self._state:
				self._cancelling -= 1
				self._state.notify_all()

	def pause(self, switch: bool) -> None:
		"""Pause playback and queued work without discarding the current request."""
		with self._playerControl:
			with self._state:
				self._isPaused = switch
				player = self._player
				self._state.notify_all()
			if player is not None:
				player.pause(switch)

	def stop(self) -> None:
		"""Interrupt the active request and request worker shutdown."""
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
		try:
			while (utterance := self._queue.get()) is not None:
				cancelled, items = utterance
				if not self._waitUntilReady(cancelled):
					continue
				try:
					self._speak(items, cancelled)
				except Exception:
					if not cancelled.is_set():
						log.exception("Error synthesizing Sonata speech")
					with self._playerControl:
						if self._player is not None:
							self._player.stop()
				# Completion follows playback and the whole queue, not each text segment.
				self._notify(cancelled)
		finally:
			try:
				with self._playerControl:
					player, self._player = self._player, None
					if player is not None:
						player.close()
			finally:
				self._client.close()

	def _notify(self, cancelled: Event, index: int | None = None) -> None:
		"""Deliver an index or queue completion without holding playback state locks."""
		with self._notifications:
			with self._state:
				if cancelled.is_set() or self._isStopping or (index is None and not self._queue.empty()):
					return
			if index is None:
				synthDoneSpeaking.notify(synth=self._synth)
			else:
				synthIndexReached.notify(synth=self._synth, index=index)

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
					self._notify(cancelled, index=item.index)
		if self._waitUntilReady(cancelled) and self._player is not None:
			self._player.idle()

	def _synthesize(self, speech: Speech, cancelled: Event) -> None:
		voice = speech.voice
		audioFormat = (voice.numChannels, voice.sampleRate, voice.sampleWidth * 8)
		stream = None
		if speech.pitch != 50 or (speech.rateBoost and speech.rate != 50):
			if not self._sonicInitialized:
				_sonic.initialize()
				self._sonicInitialized = True
			# Each segment owns its buffer. Cancellation or failure destroys any tail
			# without flushing it into the next utterance or across an index boundary.
			stream = _sonic.SonicStream(voice.sampleRate, voice.numChannels)
			stream.pitch = 2.0 ** ((speech.pitch - 50) / 50)
			if speech.rateBoost:
				# Keep the lower half of the rate slider unchanged, with normal speed
				# at 50 and up to six times normal speed at 100.
				base = 2.0 if speech.rate <= 50 else 6.0
				stream.speed = base ** ((speech.rate - 50) / 50)
		for data in self._client.synthesize(
			voice,
			speech.text,
			speaker=speech.speaker,
			rate=50 if speech.rateBoost else speech.rate,
			volume=speech.volume,
			cancelled=cancelled,
		):
			if not self._waitUntilReady(cancelled):
				return
			if stream is not None:
				# The client validates complete PCM16 frames before yielding them.
				buffer = (c_short * (len(data) // 2)).from_buffer_copy(data)
				stream.writeShort(buffer, len(buffer) // voice.numChannels)
				data = bytes(stream.readShort())
			if data:
				self._feed(data, audioFormat, cancelled)
		if stream is not None and self._waitUntilReady(cancelled):
			# Drain Sonic before _speak can sync playback and report an NVDA index.
			stream.flush()
			if data := bytes(stream.readShort()):
				self._feed(data, audioFormat, cancelled)

	def _silence(self, silence: Silence, cancelled: Event) -> None:
		# Limit buffer size so long breaks remain interruptible.
		remaining = silence.sampleRate * silence.duration // 1000
		while remaining > 0 and self._waitUntilReady(cancelled):
			samples = min(remaining, max(1, silence.sampleRate // 20))
			self._feed(
				bytes(samples * silence.numChannels * silence.sampleWidth),
				(silence.numChannels, silence.sampleRate, silence.sampleWidth * 8),
				cancelled,
				isSilence=True,
			)
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
			if not self._waitUntilReady(cancelled):
				return
			with self._playerControl:
				oldPlayer, self._player = self._player, None
				oldPlayer.close()
		with self._playerControl:
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
		# A native feed may have started just after stop; discard its stale audio.
		if cancelled.is_set():
			with self._playerControl:
				player.stop()
