import threading
import queue
from typing import Optional, Dict, Any

import nvwave
from logHandler import log
from synthDriverHandler import synthIndexReached, synthDoneSpeaking

try:
	from piper import PiperVoice
	from piper.config import SynthesisConfig
	PIPER_AVAILABLE = True
except ImportError:
	PiperVoice = None
	SynthesisConfig = None
	PIPER_AVAILABLE = False
	log.warning("Piper TTS package not installed. Install with: pip install piper-tts")


def isPiperAvailable() -> bool:
	"""Check if the Piper TTS package is available."""
	return PIPER_AVAILABLE


def loadVoice(
	model_path: str,
	config_path: str,
	use_cuda: bool = False,
	use_persian_phonemizer: bool = False,
	ezafe_model_path: Optional[str] = None,
) -> Optional[Any]:
	if not PIPER_AVAILABLE:
		log.error("Cannot load voice: Piper TTS package not available")
		return None

	try:
		voice = PiperVoice.load(
			model_path=model_path,
			config_path=config_path,
			use_cuda=use_cuda,
			use_persian_phonemizer=use_persian_phonemizer,
			ezafe_model_path=ezafe_model_path,
		)
		return voice
	except Exception as e:
		log.error(f"Failed to load Piper voice: {e}")
		return None


class BgThread(threading.Thread):
	def __init__(self, synth):
		super().__init__(daemon=True)
		self._synth = synth
		self._queue: queue.Queue = queue.Queue()
		self._running = True
		self._player: Optional[nvwave.WavePlayer] = None

	def run(self):
		while self._running:
			try:
				item = self._queue.get(timeout=0.1)
			except queue.Empty:
				continue

			if item is None:
				# Shutdown signal
				break

			try:
				self._processItem(item)
			except Exception as e:
				log.error(f"Piper synthesis error: {e}")

		self._closePlayer()

	def _processItem(self, item):
		if isinstance(item, tuple):
			cmd, data = item
			if cmd == "speak":
				self._speak(data)
			elif cmd == "index":
				synthIndexReached.notify(synth=self._synth, index=data)
			elif cmd == "cancel":
				self._cancel()

	def _speak(self, data: Dict[str, Any]):
		text = data.get("text", "")
		if not text.strip():
			return

		voice = self._synth._currentVoice
		if voice is None:
			return

		if not PIPER_AVAILABLE or SynthesisConfig is None:
			log.error("Cannot synthesize: Piper TTS package not available")
			return

		syn_config = SynthesisConfig(
			speaker_id=data.get("speaker_id"),
			length_scale=data.get("length_scale", 1.0),
			noise_scale=data.get("noise_scale", 0.667),
			noise_w_scale=data.get("noise_w_scale", 0.8),
			volume=data.get("volume", 1.0),
		)

		try:
			for audio_chunk in voice.synthesize(text, syn_config):
				if not self._running:
					break

				if self._player is None or self._player._sampleRate != audio_chunk.sample_rate:
					self._closePlayer()
					self._player = nvwave.WavePlayer(
						channels=audio_chunk.sample_channels,
						samplesPerSec=audio_chunk.sample_rate,
						bitsPerSample=audio_chunk.sample_width * 8,
					)

				self._player.feed(audio_chunk.audio_int16_bytes)

			if self._player:
				self._player.idle()

		except Exception as e:
			log.error(f"Piper synthesis failed: {e}")

		synthDoneSpeaking.notify(synth=self._synth)

	def _cancel(self):
		try:
			while True:
				self._queue.get_nowait()
		except queue.Empty:
			pass

		if self._player:
			self._player.stop()

	def _closePlayer(self):
		if self._player:
			try:
				self._player.close()
			except Exception:
				pass
			self._player = None

	def queueSpeak(self, text: str, **kwargs):
		self._queue.put(("speak", {"text": text, **kwargs}))

	def queueIndex(self, index: int):
		self._queue.put(("index", index))

	def queueCancel(self):
		self._queue.put(("cancel", None))

	def stop(self):
		self._running = False
		self._queue.put(None)
