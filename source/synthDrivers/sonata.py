# A part of NonVisual Desktop Access (NVDA)
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""NVDA driver for a separately running, local Sonata-compatible speech service."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from importlib.util import find_spec

import config
import globalVars
import languageHandler
from autoSettingsUtils.driverSetting import BooleanDriverSetting
from autoSettingsUtils.utils import StringParameterInfo
from logHandler import log
from speech.commands import (
	BreakCommand,
	IndexCommand,
	LangChangeCommand,
	PitchCommand,
	RateCommand,
	VolumeCommand,
)
from speech.types import SpeechSequence
from synthDriverHandler import SynthDriver as BaseSynthDriver
from synthDriverHandler import VoiceInfo, synthDoneSpeaking, synthIndexReached

from ._sonata import worker
from ._sonata.client import Client, Voice
from ._sonata.text import languageRuns

_CONNECTION_CONFIG_SPEC = {
	"endpoint": "string(default='127.0.0.1:50051')",
	"voicePaths": "string_list(default=list('default'))",
	"requestTimeout": "float(min=1, max=300, default=30)",
}


def _getConnectionSettings() -> tuple[str, list[str], float]:
	"""Read advanced connection options, allowing explicit environment overrides."""
	speechConfig = config.conf["speech"]
	if speechConfig.isSet("sonata"):
		section = speechConfig["sonata"]
		section.spec.update(_CONNECTION_CONFIG_SPEC)
		endpoint = section["endpoint"]
		voicePaths = section["voicePaths"]
		timeout = section["requestTimeout"]
	else:
		endpoint, voicePaths, timeout = "127.0.0.1:50051", ["default"], 30.0
	endpoint = os.environ.get("NVDA_SONATA_ENDPOINT", endpoint)
	if "NVDA_SONATA_VOICES" in os.environ:
		voicePaths = json.loads(os.environ["NVDA_SONATA_VOICES"])
	if not isinstance(endpoint, str) or not endpoint.strip():
		raise ValueError("Sonata endpoint must be a nonempty host:port string")
	if (
		not isinstance(voicePaths, list)
		or not voicePaths
		or any(not isinstance(path, str) or not path.strip() for path in voicePaths)
	):
		raise ValueError("Sonata voicePaths must be a nonempty list of voice aliases or paths")
	if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= 300:
		raise ValueError("Sonata requestTimeout must be between 1 and 300 seconds")
	return endpoint, voicePaths, float(timeout)


class SynthDriver(BaseSynthDriver):
	"""Play service-produced PCM through NVDA without importing a TTS engine."""

	name = "sonata"
	# Translators: Description for a synthesizer connecting to a local speech service.
	description = _("Sonata speech service")
	supportedSettings = (
		BaseSynthDriver.VoiceSetting(),
		BaseSynthDriver.VariantSetting(),
		BaseSynthDriver.RateSetting(),
		BaseSynthDriver.RateBoostSetting(),
		BaseSynthDriver.PitchSetting(),
		BaseSynthDriver.VolumeSetting(),
		BooleanDriverSetting(
			"detectLanguage",
			# Translators: Switch between loaded Persian and English voices for text without language markup.
			_("Detect Persian and English text"),
		),
	)
	supportedCommands = frozenset(
		{IndexCommand, BreakCommand, LangChangeCommand, PitchCommand, RateCommand, VolumeCommand},
	)
	supportedNotifications = frozenset({synthIndexReached, synthDoneSpeaking})

	@classmethod
	def check(cls) -> bool:
		"""Discover the driver without importing inference engines or contacting a server."""
		if globalVars.appArgs.secure:
			return False
		try:
			return find_spec("grpc") is not None and find_spec("google.protobuf") is not None
		except (ImportError, ValueError):
			return False

	def __init__(self) -> None:
		self._worker: worker.SpeechWorker | None = None
		self._client: Client | None = None
		self._voiceData: dict[str, Voice] = {}
		self._currentVoiceId = ""
		self._rate = 50
		self._rateBoost = False
		self._pitch = 50
		self._volume = 100
		self._variant = ""
		self._speakerByVoice: dict[str, str] = {}
		self._detectLanguage = False
		if globalVars.appArgs.secure:
			raise RuntimeError("Sonata speech service is unavailable in secure mode")
		endpoint, voicePaths, timeout = _getConnectionSettings()
		client = Client(endpoint, timeout=timeout)
		registered = False
		try:
			voices = client.loadVoices(voicePaths, timeout=min(timeout, 2.0))
			self._voiceData = {voice.id: voice for voice in voices}
			if not self._voiceData:
				raise RuntimeError("The Sonata service returned no voices")
			self._selectInitialVoice()
			# Register configuration callbacks only after metadata loading succeeds.
			super().__init__()
			registered = True
			self._client = client
			self._worker = worker.SpeechWorker(self, config.conf["audio"]["outputDevice"], client)
			self._worker.start()
		except BaseException:
			self._worker = None
			self._client = None
			client.close()
			if registered:
				self._unregisterConfigSaveAction()
			raise

	def getConfigSpec(self) -> dict[str, str]:
		spec = super().getConfigSpec()
		spec.update(_CONNECTION_CONFIG_SPEC)
		return spec

	def terminate(self) -> None:
		"""Cancel requests and join playback before saving and releasing driver settings."""
		try:
			if self._worker is not None:
				try:
					self._worker.stop()
				finally:
					self._worker.join()
				self._worker = None
			elif self._client is not None:
				self._client.close()
		finally:
			self._client = None
			try:
				super().terminate()
			finally:
				self._voiceData.clear()

	def _selectInitialVoice(self) -> None:
		language = languageHandler.getLanguage()
		languages = {
			voiceId: languageHandler.normalizeLanguage(info.language) if info.language else None
			for voiceId, info in self._voiceData.items()
		}
		voiceId = min(
			self._voiceData,
			key=lambda voiceId: (
				languages[voiceId] != language,
				(languages[voiceId] or "").split("_")[0] != language.split("_")[0],
			),
		)
		self._set_voice(voiceId)

	def _get_voice(self) -> str:
		return self._currentVoiceId

	def _set_voice(self, voiceId: str) -> None:
		if voiceId not in self._voiceData:
			raise ValueError(f"Unknown Sonata voice: {voiceId}")
		if voiceId == self._currentVoiceId:
			return
		if self._currentVoiceId:
			self._speakerByVoice[self._currentVoiceId] = self._variant
		self._currentVoiceId = voiceId
		self.__dict__.pop("_availableVariants", None)
		previousSpeaker = self._speakerByVoice.get(voiceId)
		self._variant = (
			previousSpeaker
			if previousSpeaker is not None and previousSpeaker in self.availableVariants
			else next(iter(self.availableVariants))
		)

	def _getAvailableVoices(self) -> OrderedDict[str, VoiceInfo]:
		return OrderedDict(
			(
				voiceId,
				VoiceInfo(
					voiceId,
					info.name,
					languageHandler.normalizeLanguage(info.language) if info.language else None,
				),
			)
			for voiceId, info in self._voiceData.items()
		)

	def _get_variant(self) -> str:
		return self._variant

	def _set_variant(self, variant: str) -> None:
		if variant not in self.availableVariants:
			raise ValueError(f"Unknown Sonata speaker: {variant}")
		self._variant = variant
		self._speakerByVoice[self._currentVoiceId] = variant

	def _getAvailableVariants(self) -> OrderedDict[str, StringParameterInfo]:
		speakers = self._voiceData[self._currentVoiceId].speakers
		if not speakers:
			# Translators: A speech voice's default speaker when no named speakers are provided.
			speakers = {"": _("Default")}
		return OrderedDict((key, StringParameterInfo(key, name)) for key, name in speakers.items())

	def _get_rate(self) -> int:
		return self._rate

	def _set_rate(self, rate: int) -> None:
		self._rate = max(0, min(100, rate))

	def _get_rateBoost(self) -> bool:
		return self._rateBoost

	def _set_rateBoost(self, enabled: bool) -> None:
		self._rateBoost = bool(enabled)

	def _get_pitch(self) -> int:
		return self._pitch

	def _set_pitch(self, pitch: int) -> None:
		self._pitch = max(0, min(100, pitch))

	def _get_detectLanguage(self) -> bool:
		return self._detectLanguage

	def _set_detectLanguage(self, enabled: bool) -> None:
		self._detectLanguage = bool(enabled)

	def _voiceForLanguage(self, language: str | None, default: Voice) -> Voice:
		"""Prefer an exact locale, then the same language; preserve the user's fallback voice."""
		if not language:
			return default
		requested = language.replace("-", "_").lower()
		candidates = [default, *(voice for voice in self._voiceData.values() if voice.id != default.id)]
		for exact in (True, False):
			for voice in candidates:
				available = (voice.language or "").replace("-", "_").lower()
				if available and (
					available == requested if exact else available.split("_")[0] == requested.split("_")[0]
				):
					return voice
		return default

	def _speakerForVoice(self, voice: Voice) -> str:
		if voice.id == self._currentVoiceId:
			return self._variant
		previous = self._speakerByVoice.get(voice.id)
		return (
			previous
			if previous is not None and previous in voice.speakers
			else next(iter(voice.speakers), "")
		)

	def _get_volume(self) -> int:
		return self._volume

	def _set_volume(self, volume: int) -> None:
		self._volume = max(0, min(100, volume))

	def speak(self, speechSequence: SpeechSequence) -> None:
		"""Queue speech and command boundaries using an immutable settings snapshot."""
		if self._worker is None:
			return
		defaultVoice = voice = self._voiceData[self._currentVoiceId]
		items: list[worker.SpeechItem] = []
		textParts: list[str] = []
		rate, volume, pitch = self._rate, self._volume, self._pitch
		explicitLanguage = False
		defaultLanguage = (defaultVoice.language or "").replace("-", "_").lower()
		baseLanguage = defaultLanguage.split("_")[0]
		detectLanguage = self._detectLanguage and baseLanguage in {"fa", "en"}

		def flushText() -> None:
			text = "".join(textParts)
			textParts.clear()
			runs = languageRuns(text) if detectLanguage and not explicitLanguage else [(text, None)]
			# Merge runs that fall back to the same voice, preserving Persian phrase context.
			segments: list[tuple[str, Voice]] = []
			for part, language in runs:
				selected = self._voiceForLanguage(language, voice)
				if segments and segments[-1][1].id == selected.id:
					segments[-1] = (segments[-1][0] + part, selected)
				else:
					segments.append((part, selected))
			for part, selected in segments:
				if part.strip():
					items.append(
						worker.Speech(
							part,
							selected,
							self._speakerForVoice(selected),
							rate,
							volume,
							pitch=pitch,
							rateBoost=self._rateBoost,
						),
					)

		for item in speechSequence:
			if isinstance(item, str):
				textParts.append(item)
				continue
			flushText()
			if isinstance(item, IndexCommand):
				items.append(worker.Index(item.index))
			elif isinstance(item, BreakCommand):
				items.append(
					worker.Silence(max(0, item.time), voice.sampleRate, voice.numChannels, voice.sampleWidth),
				)
			elif isinstance(item, RateCommand):
				rate = self._rate if item.isDefault else max(0, min(100, item.newValue))
			elif isinstance(item, VolumeCommand):
				volume = self._volume if item.isDefault else max(0, min(100, item.newValue))
			elif isinstance(item, PitchCommand):
				pitch = self._pitch if item.isDefault else max(0, min(100, item.newValue))
			elif isinstance(item, LangChangeCommand):
				# NVDA also prefixes ordinary, unmarked text with its default language.
				# Permit opt-in detection there; honor commands selecting another language.
				requested = (item.lang or "").replace("-", "_").lower()
				explicitLanguage = bool(requested and requested not in {defaultLanguage, baseLanguage})
				voice = self._voiceForLanguage(item.lang, defaultVoice)
			else:
				log.debugWarning(f"Unsupported Sonata speech command: {type(item).__name__}")
		flushText()
		self._worker.speak(items)

	def cancel(self) -> None:
		if self._worker is not None:
			self._worker.cancel()

	def pause(self, switch: bool) -> None:
		if self._worker is not None:
			self._worker.pause(switch)
