# A part of NonVisual Desktop Access (NVDA)
# This file is covered by the GNU General Public License.
# See the file COPYING for more details.

"""Cancellable Sonata gRPC client without dependencies on NVDA or a TTS engine."""

from __future__ import annotations

import ipaddress
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from threading import Event, Lock
from urllib.parse import urlsplit

MAX_AUDIO_MESSAGE = 16 * 1024 * 1024
MAX_TEXT_BYTES = 256 * 1024
SERVICE = "/sonata_grpc.sonata_grpc/"


@dataclass(frozen=True)
class Voice:
	"""Server metadata with a stable configured identifier for NVDA's saved settings."""

	id: str
	remoteId: str
	name: str
	language: str | None
	sampleRate: int
	numChannels: int
	sampleWidth: int
	speakers: dict[str, str]
	defaultLengthScale: float = 1.0


def validateEndpoint(endpoint: str) -> str:
	"""Restrict this unencrypted transport to literal loopback addresses."""
	try:
		parsed = urlsplit("//" + endpoint)
		host, port = parsed.hostname, parsed.port
		if (
			not host
			or port is None
			or not 1 <= port <= 65535
			or parsed.username is not None
			or parsed.password is not None
			or parsed.path
			or parsed.query
			or parsed.fragment
		):
			raise ValueError
		if host == "localhost":
			host = "127.0.0.1"
		address = ipaddress.ip_address(host)
		if not address.is_loopback:
			raise ValueError
		return f"[{address}]:{port}" if address.version == 6 else f"{address}:{port}"
	except (TypeError, ValueError) as error:
		raise ValueError("Sonata requires a loopback address and port, such as 127.0.0.1:50051") from error


class Client:
	"""Own one channel and one active request; cancellation never waits for the server."""

	def __init__(self, endpoint: str, timeout: float = 30) -> None:
		endpoint = validateEndpoint(endpoint)
		if not math.isfinite(timeout) or not 1 <= timeout <= 300:
			raise ValueError("Sonata request timeout must be between 1 and 300 seconds")
		# Driver discovery must work even if the transport dependencies are unavailable.
		import grpc

		from . import sonata_grpc_pb2 as messages

		self._grpc = grpc
		self._messages = messages
		self._timeout = timeout
		self._state = Lock()
		self._active = None
		self._closed = False
		self._remoteIds: dict[str, str] = {}
		self._channel = grpc.insecure_channel(
			endpoint,
			options=(
				("grpc.enable_http_proxy", 0),
				("grpc.max_receive_message_length", MAX_AUDIO_MESSAGE),
				("grpc.max_send_message_length", MAX_TEXT_BYTES + 4096),
			),
		)
		self._load = self._channel.unary_unary(
			SERVICE + "LoadVoice",
			request_serializer=messages.VoicePath.SerializeToString,
			response_deserializer=messages.VoiceInfo.FromString,
		)
		self._options = self._channel.unary_unary(
			SERVICE + "SetSynthesisOptions",
			request_serializer=messages.VoiceSynthesisOptions.SerializeToString,
			response_deserializer=messages.SynthesisOptions.FromString,
		)
		self._speak = self._channel.unary_stream(
			SERVICE + "SynthesizeUtterance",
			request_serializer=messages.Utterance.SerializeToString,
			response_deserializer=messages.SynthesisResult.FromString,
		)

	def _register(self, call, cancelled: Event) -> bool:
		with self._state:
			if not self._closed and not cancelled.is_set():
				self._active = call
				return True
		call.cancel()
		return False

	def _release(self, call) -> None:
		with self._state:
			if self._active is call:
				self._active = None

	def _unary(self, method, request, cancelled: Event, timeout: float):
		call = method.future(request, timeout=max(0.001, timeout))
		try:
			self._register(call, cancelled)
			return call.result()
		finally:
			self._release(call)

	@staticmethod
	def _voice(path: str, info) -> Voice:
		audio = info.audio
		if (
			not info.voice_id
			or len(info.voice_id) > 4096
			or not 8000 <= audio.sample_rate <= 192000
			or audio.num_channels not in (1, 2)
			or audio.sample_width != 2
			or len(info.speakers) > 1024
			or any(key < 0 for key in info.speakers)
		):
			raise ValueError("Sonata returned invalid voice metadata or unsupported audio (requires PCM16)")
		lengthScale = info.synth_options.length_scale if info.synth_options.HasField("length_scale") else 1.0
		if not math.isfinite(lengthScale) or lengthScale <= 0:
			raise ValueError("Sonata returned an invalid default speaking speed")
		name = path.replace("\\", "/").rsplit("/", 1)[-1]
		for suffix in (".onnx.json", ".onnx"):
			if name.endswith(suffix):
				name = name[: -len(suffix)]
				break
		return Voice(
			id=path,
			remoteId=info.voice_id,
			name=name,
			language=info.language if info.HasField("language") else None,
			sampleRate=audio.sample_rate,
			numChannels=audio.num_channels,
			sampleWidth=audio.sample_width,
			speakers={str(key): value for key, value in sorted(info.speakers.items())} or {"0": "Default"},
			defaultLengthScale=lengthScale,
		)

	def loadVoices(self, paths: list[str], timeout: float = 2) -> list[Voice]:
		"""Load configured server aliases/paths within one total startup deadline."""
		if not paths or len(paths) > 32 or any(not isinstance(path, str) or not path for path in paths):
			raise ValueError("Configure between 1 and 32 nonempty Sonata voice paths")
		deadline = time.monotonic() + timeout
		voices = []
		for path in dict.fromkeys(paths):
			remaining = deadline - time.monotonic()
			if remaining <= 0:
				raise TimeoutError("Timed out loading Sonata voices")
			info = self._unary(self._load, self._messages.VoicePath(config_path=path), Event(), remaining)
			voice = self._voice(path, info)
			self._remoteIds[path] = voice.remoteId
			voices.append(voice)
		return voices

	def synthesize(
		self,
		voice: Voice,
		text: str,
		speaker: str,
		rate: int,
		volume: int,
		cancelled: Event,
	) -> Iterator[bytes]:
		"""Stream PCM; allow a stale handle to reload once before producing any audio."""
		if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
			raise ValueError("Sonata speech segment exceeds the request size limit")
		if speaker not in voice.speakers or not 0 <= rate <= 100 or not 0 <= volume <= 100:
			raise ValueError("Invalid Sonata speech settings")
		messages = self._messages
		# Sonata's wire rate maps 0..100 to speed 0.5..5.5 (10 is normal speed).
		# Preserve NVDA's useful half/normal/double speed range at 0/50/100.
		wireRate = round((2.0 ** ((rate - 50) / 50) - 0.5) * 20)
		deadline = time.monotonic() + self._timeout
		for attempt in range(2):
			if cancelled.is_set() or self._closed:
				return
			stream = None
			producedAudio = False
			try:
				remoteId = self._remoteIds.get(voice.id, voice.remoteId)
				self._unary(
					self._options,
					messages.VoiceSynthesisOptions(
						voice_id=remoteId,
						synthesis_options=messages.SynthesisOptions(speaker=voice.speakers[speaker]),
					),
					cancelled,
					deadline - time.monotonic(),
				)
				if cancelled.is_set() or self._closed:
					return
				stream = self._speak(
					messages.Utterance(
						voice_id=remoteId,
						text=text,
						speech_args=messages.SpeechArgs(rate=wireRate, volume=volume),
						synthesis_mode=messages.MODE_LAZY,
					),
					timeout=max(0.001, deadline - time.monotonic()),
				)
				if not self._register(stream, cancelled):
					return
				for response in stream:
					if cancelled.is_set() or self._closed:
						return
					data = response.wav_samples
					if len(data) % (voice.numChannels * voice.sampleWidth):
						raise ValueError("Sonata returned an incomplete PCM sample frame")
					if data:
						producedAudio = True
						# Bound audio feeds even when upstream sends a whole sentence per response.
						for start in range(0, len(data), 65536):
							if cancelled.is_set() or self._closed:
								return
							yield data[start : start + 65536]
				return
			except (self._grpc.RpcError, self._grpc.FutureCancelledError) as error:
				if cancelled.is_set() or self._closed:
					return
				if (
					attempt == 0
					and not producedAudio
					and isinstance(error, self._grpc.RpcError)
					and error.code() == self._grpc.StatusCode.NOT_FOUND
				):
					info = self._unary(
						self._load,
						messages.VoicePath(config_path=voice.id),
						cancelled,
						deadline - time.monotonic(),
					)
					updated = self._voice(voice.id, info)
					if (updated.sampleRate, updated.numChannels, updated.sampleWidth, updated.speakers) != (
						voice.sampleRate,
						voice.numChannels,
						voice.sampleWidth,
						voice.speakers,
					):
						raise ValueError("Sonata voice changed after reconnect; reload the synthesizer")
					self._remoteIds[voice.id] = updated.remoteId
					continue
				raise
			finally:
				if stream is not None:
					stream.cancel()
					self._release(stream)

	def cancel(self) -> None:
		with self._state:
			active = self._active
		if active is not None:
			active.cancel()

	def close(self) -> None:
		with self._state:
			if self._closed:
				return
			self._closed = True
		self.cancel()
		self._channel.close()
