# A part of NonVisual Desktop Access (NVDA)
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""A loopback Sonata gRPC frontend for the custom Piper Python engine."""

from __future__ import annotations

import argparse
import logging
import math
import os
import threading
import time
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import grpc

from . import __version__
from . import sonata_grpc_pb2 as messages
from .backend import Settings, Voice, loadPiperVoice, localVoicePaths

LOG = logging.getLogger(__name__)
MAX_TEXT_BYTES = 256 * 1024
MAX_AUDIO_BYTES = 64 * 1024
MAX_MESSAGE_BYTES = MAX_TEXT_BYTES + 4096
MAX_SESSIONS = 1024
SESSION_IDLE_SECONDS = 24 * 60 * 60
SERVICE_NAME = "sonata_grpc.sonata_grpc"


@dataclass
class _Session:
	voice: Voice
	settings: Settings
	lastUsed: float


def _settingsMessage(settings: Settings, voice: Voice) -> messages.SynthesisOptions:
	result = messages.SynthesisOptions(
		length_scale=settings.lengthScale,
		noise_scale=settings.noiseScale,
		noise_w=settings.noiseW,
	)
	if settings.speaker is not None:
		result.speaker = voice.speakers[settings.speaker]
	return result


class PiperService:
	"""Share resident models while keeping options private to each LoadVoice handle."""

	def __init__(self, voices: dict[str, Voice]):
		if not voices:
			raise ValueError("At least one configured voice is required")
		self._voices = dict(voices)
		self._sessions: dict[str, _Session] = {}
		self._stateLock = threading.Lock()
		self._voiceLocks = {id(voice): threading.Lock() for voice in voices.values()}

	def _session(self, voiceId: str, context) -> _Session:
		# The caller owns _stateLock, including when replacing session settings.
		try:
			session = self._sessions[voiceId]
		except KeyError:
			context.abort(grpc.StatusCode.NOT_FOUND, "Unknown voice handle; call LoadVoice first")
		session.lastUsed = time.monotonic()
		return session

	def _voiceInfo(self, voiceId: str, session: _Session) -> messages.VoiceInfo:
		return messages.VoiceInfo(
			voice_id=voiceId,
			synth_options=_settingsMessage(session.settings, session.voice),
			speakers=session.voice.speakers,
			audio=messages.AudioInfo(sample_rate=session.voice.sampleRate, num_channels=1, sample_width=2),
			language=session.voice.language,
			# Piper yields completed sentences. It does not implement Sonata's realtime RPC.
			supports_streaming_output=False,
		)

	def GetSonataVersion(self, request, context):
		return messages.Version(version=f"sonata-piper-service/{__version__}")

	def LoadVoice(self, request, context):
		# Only lookup preconfigured aliases. Network input never becomes a filesystem operation.
		alias = os.path.normcase(request.config_path)
		voice = self._voices.get(alias)
		if voice is None:
			context.abort(grpc.StatusCode.NOT_FOUND, "Voice is not configured on this service")
		with self._stateLock:
			now = time.monotonic()
			# The protocol has no release RPC. Retire old handles only when the bounded table is full.
			if len(self._sessions) >= MAX_SESSIONS:
				self._sessions = {
					key: value
					for key, value in self._sessions.items()
					if now - value.lastUsed < SESSION_IDLE_SECONDS
				}
			if len(self._sessions) >= MAX_SESSIONS:
				context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "Too many voice handles")
			voiceId = uuid.uuid4().hex
			session = _Session(voice, voice.defaults, now)
			self._sessions[voiceId] = session
			return self._voiceInfo(voiceId, session)

	def GetVoiceInfo(self, request, context):
		with self._stateLock:
			return self._voiceInfo(request.voice_id, self._session(request.voice_id, context))

	def GetSynthesisOptions(self, request, context):
		with self._stateLock:
			session = self._session(request.voice_id, context)
			return _settingsMessage(session.settings, session.voice)

	def SetSynthesisOptions(self, request, context):
		with self._stateLock:
			session = self._session(request.voice_id, context)
			incoming = request.synthesis_options
			updates = {}
			if incoming.HasField("speaker"):
				# Upstream Sonata resolves the speaker's NAME to its numeric model ID.
				speakersByName = {name: speakerId for speakerId, name in session.voice.speakers.items()}
				if incoming.speaker not in speakersByName:
					context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Unknown speaker name")
				updates["speaker"] = speakersByName[incoming.speaker]
			for field, setting, minimum in (
				("length_scale", "lengthScale", 0.1),
				("noise_scale", "noiseScale", 0),
				("noise_w", "noiseW", 0),
			):
				if incoming.HasField(field):
					value = getattr(incoming, field)
					if not math.isfinite(value) or not minimum <= value <= 10:
						context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Invalid {field}")
					updates[setting] = value
			# Validate all fields before changing any options.
			session.settings = replace(session.settings, **updates)
			return _settingsMessage(session.settings, session.voice)

	def SynthesizeUtterance(self, request, context):
		if len(request.text.encode("utf-8")) > MAX_TEXT_BYTES:
			context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Text exceeds 256 KiB")
		if request.synthesis_mode not in (messages.MODE_UNSPECIFIED, messages.MODE_LAZY):
			context.abort(grpc.StatusCode.UNIMPLEMENTED, "Only lazy synthesis is supported")
		args = request.speech_args
		rate = args.rate if args.HasField("rate") else 10
		volume = args.volume if args.HasField("volume") else 100
		if rate > 100 or volume > 100:
			context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Rate and volume must be between 0 and 100")
		if args.HasField("pitch") and args.pitch != 50:
			context.abort(grpc.StatusCode.UNIMPLEMENTED, "Piper does not support pitch adjustment")
		if args.appended_silence_ms > 60000:
			context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Appended silence exceeds 60 seconds")
		with self._stateLock:
			session = self._session(request.voice_id, context)
			voice = session.voice
			settings = replace(session.settings, lengthScale=session.settings.lengthScale / (0.5 + rate / 20))
		if not request.text.strip():
			return

		def cancelled():
			return not context.is_active()

		voiceLock = self._voiceLocks[id(voice)]
		# A cancelled request waiting for the resident model never starts inference.
		while not voiceLock.acquire(timeout=0.05):
			if cancelled():
				return
		try:
			if cancelled():
				return
			for audio in voice.synthesize(request.text, settings, volume / 100, cancelled):
				if cancelled():
					return
				if (
					audio.sampleRate != voice.sampleRate
					or audio.channels != 1
					or audio.width != 2
					or len(audio.data) % 2
				):
					raise ValueError("Engine produced audio outside the advertised PCM16LE format")
				for offset in range(0, len(audio.data), MAX_AUDIO_BYTES):
					if cancelled():
						return
					yield messages.SynthesisResult(wav_samples=audio.data[offset : offset + MAX_AUDIO_BYTES])
		except Exception:
			# Do not put engine exceptions, local paths, or spoken text into network error messages.
			LOG.exception("Piper synthesis failed")
			if not cancelled():
				context.abort(grpc.StatusCode.INTERNAL, "Piper synthesis failed; see service log")
			return
		finally:
			voiceLock.release()
		remaining = voice.sampleRate * args.appended_silence_ms // 1000 * 2
		while remaining and not cancelled():
			size = min(remaining, MAX_AUDIO_BYTES)
			yield messages.SynthesisResult(wav_samples=bytes(size))
			remaining -= size

	def SynthesizeUtteranceRealtime(self, request, context):
		context.abort(
			grpc.StatusCode.UNIMPLEMENTED,
			"Use SynthesizeUtterance; realtime inference is unavailable",
		)


def createServer(service: PiperService, *, workers: int = 8) -> grpc.Server:
	"""Register the exact upstream method paths without generated gRPC stubs."""
	server = grpc.server(
		ThreadPoolExecutor(max_workers=workers, thread_name_prefix="piper-grpc"),
		maximum_concurrent_rpcs=workers,
		options=(
			("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
			("grpc.max_send_message_length", MAX_AUDIO_BYTES + 4096),
			("grpc.so_reuseport", 0),
		),
	)
	handlers = {}
	for name, requestType, responseType, streaming in (
		("GetSonataVersion", messages.Empty, messages.Version, False),
		("LoadVoice", messages.VoicePath, messages.VoiceInfo, False),
		("GetVoiceInfo", messages.VoiceIdentifier, messages.VoiceInfo, False),
		("GetSynthesisOptions", messages.VoiceIdentifier, messages.SynthesisOptions, False),
		("SetSynthesisOptions", messages.VoiceSynthesisOptions, messages.SynthesisOptions, False),
		("SynthesizeUtterance", messages.Utterance, messages.SynthesisResult, True),
		("SynthesizeUtteranceRealtime", messages.Utterance, messages.WaveSamples, True),
	):
		factory = grpc.unary_stream_rpc_method_handler if streaming else grpc.unary_unary_rpc_method_handler
		handlers[name] = factory(
			getattr(service, name),
			request_deserializer=requestType.FromString,
			response_serializer=responseType.SerializeToString,
		)
	server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(SERVICE_NAME, handlers),))
	return server


def configuredVoices(paths: Iterable[str], **options) -> dict[str, Voice]:
	"""Preload models once and expose only aliases explicitly derived from CLI assets."""
	voices: dict[str, Voice] = {}
	models: dict[Path, Voice] = {}
	for path in paths:
		modelPath, configPath = localVoicePaths(path)
		voice = models.get(modelPath)
		if voice is None:
			LOG.info("Loading voice %s", modelPath)
			voice = models[modelPath] = loadPiperVoice(modelPath, configPath, **options)
		aliases = [modelPath.stem, modelPath.name, configPath.name, str(modelPath), str(configPath)]
		if not voices:
			aliases.append("default")
		for alias in aliases:
			alias = os.path.normcase(alias)
			if alias in voices and voices[alias] is not voice:
				raise ValueError(f"Duplicate voice alias: {alias}")
			voices[alias] = voice
	return voices


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--voice",
		action="append",
		required=True,
		help="Local .onnx or .onnx.json; repeat for more voices",
	)
	parser.add_argument("--port", type=int, default=50051, help="Loopback TCP port (default: 50051)")
	parser.add_argument(
		"--persian-phonemizer",
		action="store_true",
		help="Enable the custom Persian frontend",
	)
	parser.add_argument("--ezafe-model", help="Local Ezafe model directory")
	parser.add_argument("--homograph-dictionary", help="Local Persian homograph dictionary file")
	parser.add_argument(
		"--short-speech-repeat",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Repeat short Persian input and extract the first copy (default: enabled); requires a prepared model",
	)
	args = parser.parse_args(argv)
	if not 1 <= args.port <= 65535:
		parser.error("--port must be between 1 and 65535")
	if not args.persian_phonemizer and (args.ezafe_model or args.homograph_dictionary):
		parser.error("Persian resources require --persian-phonemizer")
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	try:
		voices = configuredVoices(
			args.voice,
			persianPhonemizer=args.persian_phonemizer,
			shortSpeechRepeat=args.short_speech_repeat,
			ezafeModel=args.ezafe_model,
			homographDictionary=args.homograph_dictionary,
		)
		server = createServer(PiperService(voices))
		address = f"127.0.0.1:{args.port}"
		if server.add_insecure_port(address) == 0:
			raise RuntimeError(f"Unable to bind {address}")
		server.start()
	except Exception:
		LOG.exception("Service startup failed")
		raise SystemExit(1) from None
	LOG.info("Piper service listening on %s; %d configured aliases", address, len(voices))
	try:
		server.wait_for_termination()
	except KeyboardInterrupt:
		LOG.info("Stopping Piper service")
	finally:
		server.stop(grace=0).wait(timeout=5)
