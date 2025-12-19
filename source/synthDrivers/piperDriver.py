import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Dict, Any

from autoSettingsUtils.driverSetting import NumericDriverSetting, BooleanDriverSetting
from autoSettingsUtils.utils import StringParameterInfo
from logHandler import log
from speech.commands import (
	IndexCommand,
	CharacterModeCommand,
	LangChangeCommand,
	BreakCommand,
	PitchCommand,
	RateCommand,
	VolumeCommand,
)
from synthDriverHandler import SynthDriver, VoiceInfo, synthIndexReached, synthDoneSpeaking
from ._piper import BgThread, isPiperAvailable, loadVoice


DEFAULT_VOICE_DIR = os.path.join(os.path.dirname(__file__), "piper_voices")


class SynthDriver(SynthDriver):
	name = "piper"
	description = "Piper TTS"

	supportedSettings = [
		SynthDriver.VoiceSetting(),
		SynthDriver.VariantSetting(),
		SynthDriver.RateSetting(),
		NumericDriverSetting(
			"pitch",
			_("&Pitch"),
			minStep=1,
		),
		SynthDriver.VolumeSetting(),
		BooleanDriverSetting(
			"usePersianPhonemizer",
			_("Use enhanced &Persian phonemizer"),
			defaultVal=True,
		),
	]

	supportedCommands = {
		IndexCommand,
		CharacterModeCommand,
		LangChangeCommand,
		BreakCommand,
		PitchCommand,
		RateCommand,
		VolumeCommand,
	}

	supportedNotifications = {synthIndexReached, synthDoneSpeaking}

	@classmethod
	def check(cls) -> bool:
		return isPiperAvailable()

	def __init__(self):
		super().__init__()

		if not isPiperAvailable():
			raise RuntimeError("Piper TTS package not available")

		self._voiceDir = Path(os.environ.get("PIPER_VOICE_DIR", DEFAULT_VOICE_DIR))
		self._ezafeModelPath = os.environ.get("PIPER_EZAFE_MODEL_PATH")

		self._availableVoices: Dict[str, Dict[str, Any]] = {}
		self._loadedVoices: Dict[str, Any] = {}
		self._currentVoice: Optional[Any] = None
		self._currentVoiceId: str = ""

		self._rate = 50
		self._pitch = 50
		self._volume = 100
		self._variant = "0"
		self._usePersianPhonemizer = True

		self._bgThread: Optional[BgThread] = None

		self._scanVoices()

		self._bgThread = BgThread(self)
		self._bgThread.start()

		if self._availableVoices:
			self.voice = next(iter(self._availableVoices.keys()))

		log.info(f"Piper TTS initialized. Voice directory: {self._voiceDir}")

	def terminate(self):
		if self._bgThread:
			self._bgThread.stop()
			self._bgThread.join(timeout=2.0)
			self._bgThread = None

		self._loadedVoices.clear()
		self._currentVoice = None

		log.info("Piper TTS terminated")

	def _scanVoices(self):
		self._availableVoices.clear()

		if not self._voiceDir.exists():
			log.warning(f"Piper voice directory does not exist: {self._voiceDir}")
			try:
				self._voiceDir.mkdir(parents=True, exist_ok=True)
			except Exception as e:
				log.error(f"Could not create voice directory: {e}")
			return

		import json

		for onnx_path in self._voiceDir.glob("*.onnx"):
			config_path = Path(f"{onnx_path}.json")
			if not config_path.exists():
				continue

			voice_id = onnx_path.stem
			try:
				with open(config_path, "r", encoding="utf-8") as f:
					config = json.load(f)

				espeak_voice = config.get("espeak", {}).get("voice", "en")
				lang = espeak_voice.split("-")[0] if "-" in espeak_voice else espeak_voice

				self._availableVoices[voice_id] = {
					"name": voice_id,
					"language": lang,
					"path": str(onnx_path),
					"config_path": str(config_path),
					"sample_rate": config.get("audio", {}).get("sample_rate", 22050),
					"num_speakers": config.get("num_speakers", 1),
					"speaker_id_map": config.get("speaker_id_map", {}),
					"espeak_voice": espeak_voice,
				}
				log.debug(f"Found Piper voice: {voice_id} ({lang})")

			except Exception as e:
				log.warning(f"Could not load voice config {config_path}: {e}")

		log.info(f"Found {len(self._availableVoices)} Piper voice(s)")

	def _loadVoice(self, voice_id: str) -> Optional[Any]:
		if voice_id in self._loadedVoices:
			return self._loadedVoices[voice_id]

		voice_info = self._availableVoices.get(voice_id)
		if not voice_info:
			log.error(f"Voice not found: {voice_id}")
			return None

		try:
			voice = loadVoice(
				model_path=voice_info["path"],
				config_path=voice_info["config_path"],
				use_cuda=False,
				use_persian_phonemizer=self._usePersianPhonemizer,
				ezafe_model_path=self._ezafeModelPath,
			)
			if voice:
				self._loadedVoices[voice_id] = voice
				log.info(f"Loaded Piper voice: {voice_id}")
			return voice

		except Exception as e:
			log.error(f"Failed to load voice {voice_id}: {e}")
			return None

	def _get_voice(self) -> str:
		return self._currentVoiceId

	def _set_voice(self, voice_id: str):
		if voice_id == self._currentVoiceId:
			return

		voice = self._loadVoice(voice_id)
		if voice:
			self._currentVoice = voice
			self._currentVoiceId = voice_id
			self._variant = "0"

	voice = property(_get_voice, _set_voice)

	def _getAvailableVoices(self) -> OrderedDict:
		"""Get available voices."""
		voices = OrderedDict()
		for voice_id, info in sorted(self._availableVoices.items()):
			voices[voice_id] = VoiceInfo(
				voice_id,
				info["name"],
				info["language"],
			)
		return voices

	availableVoices = property(_getAvailableVoices)

	def _get_variant(self) -> str:
		return self._variant

	def _set_variant(self, variant: str):
		self._variant = variant

	variant = property(_get_variant, _set_variant)

	def _getAvailableVariants(self) -> OrderedDict:
		variants = OrderedDict()

		if not self._currentVoiceId:
			return variants

		voice_info = self._availableVoices.get(self._currentVoiceId, {})
		num_speakers = voice_info.get("num_speakers", 1)
		speaker_id_map = voice_info.get("speaker_id_map", {})

		if speaker_id_map:
			for name, sid in sorted(speaker_id_map.items(), key=lambda x: x[1]):
				variants[str(sid)] = StringParameterInfo(str(sid), name)
		else:
			for i in range(num_speakers):
				variants[str(i)] = StringParameterInfo(str(i), f"Speaker {i}")

		return variants

	availableVariants = property(_getAvailableVariants)

	def _get_rate(self) -> int:
		return self._rate

	def _set_rate(self, rate: int):
		self._rate = max(0, min(100, rate))

	rate = property(_get_rate, _set_rate)

	def _get_pitch(self) -> int:
		return self._pitch

	def _set_pitch(self, pitch: int):
		self._pitch = max(0, min(100, pitch))

	pitch = property(_get_pitch, _set_pitch)

	def _get_volume(self) -> int:
		return self._volume

	def _set_volume(self, volume: int):
		self._volume = max(0, min(100, volume))

	volume = property(_get_volume, _set_volume)

	def _get_usePersianPhonemizer(self) -> bool:
		return self._usePersianPhonemizer

	def _set_usePersianPhonemizer(self, value: bool):
		self._usePersianPhonemizer = value
		if self._currentVoiceId:
			if self._currentVoiceId in self._loadedVoices:
				del self._loadedVoices[self._currentVoiceId]
			self._currentVoice = self._loadVoice(self._currentVoiceId)

	usePersianPhonemizer = property(_get_usePersianPhonemizer, _set_usePersianPhonemizer)

	def _rateToLengthScale(self, rate: int) -> float:
		# rate 0 -> length_scale 2.0 (slowest)
		# rate 50 -> length_scale 1.0 (normal)
		# rate 100 -> length_scale 0.5 (fastest)
		return 2.0 - (rate / 100.0 * 1.5)

	def _volumeToFloat(self, volume: int) -> float:
		return volume / 100.0

	def speak(self, speechSequence: list):
		if not self._bgThread or not self._currentVoice:
			return

		textParts = []
		currentRate = self._rate
		currentVolume = self._volume

		for item in speechSequence:
			if isinstance(item, str):
				textParts.append(item)

			elif isinstance(item, IndexCommand):
				if textParts:
					text = "".join(textParts)
					if text.strip():
						self._bgThread.queueSpeak(
							text,
							speaker_id=int(self._variant) if self._variant.isdigit() else 0,
							length_scale=self._rateToLengthScale(currentRate),
							volume=self._volumeToFloat(currentVolume),
						)
					textParts.clear()
				self._bgThread.queueIndex(item.index)

			elif isinstance(item, RateCommand):
				if item.isDefault:
					currentRate = self._rate
				else:
					currentRate = max(0, min(100, item.newValue))

			elif isinstance(item, VolumeCommand):
				if item.isDefault:
					currentVolume = self._volume
				else:
					currentVolume = max(0, min(100, item.newValue))

			elif isinstance(item, BreakCommand):
				# Add silence marker
				textParts.append(" ")

			elif isinstance(item, CharacterModeCommand):
				# Character mode - spell out letters
				pass

			elif isinstance(item, LangChangeCommand):
				# Language change - Piper handles this per-voice
				pass

		if textParts:
			text = "".join(textParts)
			if text.strip():
				self._bgThread.queueSpeak(
					text,
					speaker_id=int(self._variant) if self._variant.isdigit() else 0,
					length_scale=self._rateToLengthScale(currentRate),
					volume=self._volumeToFloat(currentVolume),
				)

	def cancel(self):
		if self._bgThread:
			self._bgThread.queueCancel()

	def pause(self, switch: bool):
		# Piper doesn't support pause natively
		# Could implement by stopping/restarting
		pass

	@property
	def language(self) -> Optional[str]:
		if self._currentVoiceId:
			voice_info = self._availableVoices.get(self._currentVoiceId, {})
			return voice_info.get("language")
		return None
