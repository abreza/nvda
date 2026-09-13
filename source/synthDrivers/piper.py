# A part of NonVisual Desktop Access (NVDA)
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""NVDA synthesizer driver for the custom Piper wheel with Persian support."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import config
import languageHandler
from autoSettingsUtils.driverSetting import BooleanDriverSetting
from autoSettingsUtils.utils import StringParameterInfo
from logHandler import log
from speech.commands import BreakCommand, IndexCommand, RateCommand, VolumeCommand
from speech.types import SpeechSequence
from synthDriverHandler import SynthDriver as BaseSynthDriver
from synthDriverHandler import VoiceInfo, synthDoneSpeaking, synthIndexReached

from . import _piper

if TYPE_CHECKING:
	from piper import PiperVoice


DEFAULT_VOICE_DIR = Path(__file__).with_name("piper_voices")


@dataclass(frozen=True)
class _Voice:
	modelPath: Path
	configPath: Path
	language: str | None
	sampleRate: int
	numSpeakers: int
	speakers: dict[str, int]


def _getVoiceDirectory() -> Path:
	return Path(os.environ.get("PIPER_VOICE_DIR") or DEFAULT_VOICE_DIR)


class SynthDriver(BaseSynthDriver):
	"""Expose local Piper voices through NVDA's standard synthesizer settings."""

	name = "piper"
	# Translators: Description for a speech synthesizer.
	description = _("Piper Neural TTS")
	supportedSettings = (
		BaseSynthDriver.VoiceSetting(),
		BaseSynthDriver.VariantSetting(),
		BaseSynthDriver.RateSetting(),
		BaseSynthDriver.VolumeSetting(),
		BooleanDriverSetting(
			"usePersianPhonemizer",
			# Translators: A voice setting enabling the custom Piper Persian phonemizer.
			_("Use enhanced &Persian phonemizer"),
			defaultVal=False,
		),
	)
	supportedCommands = frozenset({IndexCommand, BreakCommand, RateCommand, VolumeCommand})
	supportedNotifications = frozenset({synthIndexReached, synthDoneSpeaking})

	@classmethod
	def check(cls) -> bool:
		"""Offer Piper only when its package and at least one local voice are present."""
		if not _piper.isPiperAvailable():
			return False
		return any(model.with_suffix(".onnx.json").is_file() for model in _getVoiceDirectory().glob("*.onnx"))

	def __init__(self) -> None:
		self._voiceDir = _getVoiceDirectory()
		self._ezafeModelPath = os.environ.get("PIPER_EZAFE_MODEL_PATH") or None
		self._voiceData: dict[str, _Voice] = {}
		self._loadedVoices: dict[tuple[str, bool], PiperVoice] = {}
		self._currentVoice: PiperVoice | None = None
		self._currentVoiceId = ""
		self._rate = 50
		self._volume = 100
		self._variant = "0"
		self._usePersianPhonemizer = False
		self._worker: _piper.SpeechWorker | None = None
		self._scanVoices()
		if not self._voiceData:
			raise RuntimeError("No valid Piper voices found")
		self._selectInitialVoice()
		# Register settings only after initialization has succeeded.
		super().__init__()
		self._worker = _piper.SpeechWorker(self, config.conf["audio"]["outputDevice"])
		self._worker.start()

	def terminate(self) -> None:
		"""Stop all work before releasing voices and saving the driver's settings."""
		try:
			if self._worker is not None:
				try:
					self._worker.stop()
				finally:
					self._worker.join()
				self._worker = None
		finally:
			try:
				super().terminate()
			finally:
				self._loadedVoices.clear()
				self._currentVoice = None

	def _scanVoices(self) -> None:
		"""Read local voice metadata without creating directories or downloading files."""
		for modelPath in sorted(self._voiceDir.glob("*.onnx")):
			configPath = modelPath.with_suffix(".onnx.json")
			if not configPath.is_file():
				continue
			try:
				metadata = json.loads(configPath.read_text(encoding="utf-8-sig"))
				sampleRate = metadata["audio"]["sample_rate"]
				numSpeakers = metadata.get("num_speakers", 1)
				speakers = metadata.get("speaker_id_map", {})
				if not isinstance(sampleRate, int) or sampleRate <= 0:
					raise ValueError("Invalid sample rate")
				if not isinstance(numSpeakers, int) or numSpeakers <= 0:
					raise ValueError("Invalid speaker count")
				if not isinstance(speakers, dict) or any(
					not isinstance(name, str) or not isinstance(sid, int) or not 0 <= sid < numSpeakers
					for name, sid in speakers.items()
				):
					raise ValueError("Invalid speaker map")
				language = metadata.get("language", {}).get("code") or metadata.get("espeak", {}).get("voice")
				if language is not None:
					language = languageHandler.normalizeLanguage(language)
				self._voiceData[modelPath.stem] = _Voice(
					modelPath,
					configPath,
					language,
					sampleRate,
					numSpeakers,
					speakers,
				)
			except (OSError, ValueError, KeyError, TypeError, AttributeError):
				log.debugWarning(f"Invalid Piper voice configuration: {configPath}", exc_info=True)

	def _selectInitialVoice(self) -> None:
		language = languageHandler.getLanguage()
		voiceIds = sorted(
			self._voiceData,
			key=lambda voiceId: (
				self._voiceData[voiceId].language != language,
				(self._voiceData[voiceId].language or "").split("_")[0] != language.split("_")[0],
			),
		)
		for voiceId in voiceIds:
			try:
				self._set_voice(voiceId)
				return
			except Exception:
				log.exception(f"Unable to load Piper voice: {voiceId}")
		raise RuntimeError("Unable to load any Piper voice")

	def _loadVoice(self, voiceId: str, usePersianPhonemizer: bool) -> PiperVoice:
		key = (voiceId, usePersianPhonemizer)
		if key not in self._loadedVoices:
			voice = self._voiceData[voiceId]
			self._loadedVoices[key] = _piper.loadVoice(
				modelPath=voice.modelPath,
				configPath=voice.configPath,
				usePersianPhonemizer=usePersianPhonemizer,
				ezafeModelPath=self._ezafeModelPath,
			)
		return self._loadedVoices[key]

	def _get_voice(self) -> str:
		return self._currentVoiceId

	def _set_voice(self, voiceId: str) -> None:
		if voiceId not in self._voiceData:
			raise ValueError(f"Unknown Piper voice: {voiceId}")
		if voiceId == self._currentVoiceId:
			return
		voice = self._loadVoice(voiceId, self._usePersianPhonemizer)
		self._currentVoice = voice
		self._currentVoiceId = voiceId
		self.__dict__.pop("_availableVariants", None)
		self._variant = next(iter(self.availableVariants))

	def _getAvailableVoices(self) -> OrderedDict[str, VoiceInfo]:
		return OrderedDict(
			(voiceId, VoiceInfo(voiceId, voiceId, info.language)) for voiceId, info in self._voiceData.items()
		)

	def _get_variant(self) -> str:
		return self._variant

	def _set_variant(self, variant: str) -> None:
		if variant not in self.availableVariants:
			raise ValueError(f"Unknown Piper speaker: {variant}")
		self._variant = variant

	def _getAvailableVariants(self) -> OrderedDict[str, StringParameterInfo]:
		voice = self._voiceData[self._currentVoiceId]
		if voice.speakers:
			return OrderedDict(
				(str(speakerId), StringParameterInfo(str(speakerId), name))
				for name, speakerId in sorted(voice.speakers.items(), key=lambda item: item[1])
			)
		return OrderedDict(
			(
				str(speakerId),
				StringParameterInfo(
					str(speakerId),
					# Translators: The numbered speaker in a Piper voice with no named speakers.
					_("Speaker {number}").format(number=speakerId + 1),
				),
			)
			for speakerId in range(voice.numSpeakers)
		)

	def _get_rate(self) -> int:
		return self._rate

	def _set_rate(self, rate: int) -> None:
		self._rate = max(0, min(100, rate))

	def _get_volume(self) -> int:
		return self._volume

	def _set_volume(self, volume: int) -> None:
		self._volume = max(0, min(100, volume))

	def _get_usePersianPhonemizer(self) -> bool:
		return self._usePersianPhonemizer

	def _set_usePersianPhonemizer(self, value: bool) -> None:
		if value == self._usePersianPhonemizer:
			return
		voice = self._loadVoice(self._currentVoiceId, value)
		self._usePersianPhonemizer = value
		self._currentVoice = voice

	@staticmethod
	def _rateToLengthScale(rate: int) -> float:
		"""Map rate 0/50/100 to half/normal/double the voice's default speed."""
		return 2.0 ** ((50 - rate) / 50)

	def speak(self, speechSequence: SpeechSequence) -> None:
		"""Queue text and commands with a consistent snapshot of the current voice."""
		if self._worker is None or self._currentVoice is None:
			return
		voice = self._currentVoice
		voiceInfo = self._voiceData[self._currentVoiceId]
		speakerId = int(self._variant) if voiceInfo.numSpeakers > 1 else None
		items: list[_piper.SpeechItem] = []
		textParts: list[str] = []
		rate, volume = self._rate, self._volume

		def flushText() -> None:
			text = "".join(textParts)
			textParts.clear()
			if text.strip():
				items.append(
					_piper.Speech(
						text,
						voice,
						speakerId,
						voice.config.length_scale * self._rateToLengthScale(rate),
						volume / 100.0,
					),
				)

		for item in speechSequence:
			if isinstance(item, str):
				textParts.append(item)
				continue
			flushText()
			if isinstance(item, IndexCommand):
				items.append(_piper.Index(item.index))
			elif isinstance(item, BreakCommand):
				items.append(_piper.Silence(max(0, item.time), voiceInfo.sampleRate))
			elif isinstance(item, RateCommand):
				rate = self._rate if item.isDefault else max(0, min(100, item.newValue))
			elif isinstance(item, VolumeCommand):
				volume = self._volume if item.isDefault else max(0, min(100, item.newValue))
			else:
				log.debugWarning(f"Unsupported Piper speech command: {type(item).__name__}")
		flushText()
		self._worker.speak(items)

	def cancel(self) -> None:
		"""Silence active speech and discard pending utterances."""
		if self._worker is not None:
			self._worker.cancel()

	def pause(self, switch: bool) -> None:
		"""Pause or resume without losing queued speech."""
		if self._worker is not None:
			self._worker.pause(switch)
