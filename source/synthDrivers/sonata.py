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
from autoSettingsUtils.utils import StringParameterInfo
from logHandler import log
from speech.commands import BreakCommand, IndexCommand, RateCommand, VolumeCommand
from speech.types import SpeechSequence
from synthDriverHandler import SynthDriver as BaseSynthDriver
from synthDriverHandler import VoiceInfo, synthDoneSpeaking, synthIndexReached

from ._sonata import worker
from ._sonata.client import Client, Voice

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
		BaseSynthDriver.VolumeSetting(),
	)
	supportedCommands = frozenset({IndexCommand, BreakCommand, RateCommand, VolumeCommand})
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
		self._volume = 100
		self._variant = ""
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
		self._currentVoiceId = voiceId
		self.__dict__.pop("_availableVariants", None)
		self._variant = next(iter(self.availableVariants))

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

	def _get_volume(self) -> int:
		return self._volume

	def _set_volume(self, volume: int) -> None:
		self._volume = max(0, min(100, volume))

	def speak(self, speechSequence: SpeechSequence) -> None:
		"""Queue speech and command boundaries using an immutable settings snapshot."""
		if self._worker is None:
			return
		voice = self._voiceData[self._currentVoiceId]
		speaker = self._variant
		items: list[worker.SpeechItem] = []
		textParts: list[str] = []
		rate, volume = self._rate, self._volume

		def flushText() -> None:
			text = "".join(textParts)
			textParts.clear()
			if text.strip():
				items.append(worker.Speech(text, voice, speaker, rate, volume))

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
