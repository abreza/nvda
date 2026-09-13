# Standalone Piper service

This service runs the custom Piper engine in its own Python environment and exposes
the Sonata gRPC interface on `127.0.0.1:50051`. The built-in NVDA Sonata driver and
other gRPC clients can use it without importing Piper into NVDA. It is a Python
implementation of Sonata's RPC interface; it does not run Sonata's Rust engine.

## Install and run

The initial backend requires the project's custom Windows x64 Piper 1.3.1 wheel.
Its `PiperVoice` API adds `use_persian_phonemizer`, `ezafe_model_path`, and
`cancelled_callback`. A stock PyPI Piper wheel is not a compatible replacement.
Keep the wheel and voice assets outside the NVDA source checkout. With Python 3.13
installed, run these commands from `extras/piperService`:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\python.exe -m pip install C:\Piper\piper_tts-1.3.1-cp39-abi3-win_amd64.whl
.\.venv\Scripts\python.exe -m sonata_piper --voice C:\Piper\voices\fa_IR-mana-medium.onnx
```

The matching `.onnx.json` must sit beside the `.onnx` file. Repeat `--voice` to
preload additional voices; use `--port` to change the loopback port. The service
loads configured models once before accepting connections and keeps them resident.
Start it independently of NVDA and leave the process running while clients use it.
Stop with Control+C.

To enable the custom Persian phonemizer, install its extra dependencies and provide
the local Ezafe model and homograph dictionary explicitly:

```powershell
.\.venv\Scripts\python.exe -m pip install ".[persian]"
.\.venv\Scripts\python.exe -m sonata_piper `
    --voice C:\Piper\voices\fa_IR-mana-medium.onnx `
    --persian-phonemizer `
    --ezafe-model C:\Piper\ezafe_model_quantized `
    --homograph-dictionary C:\Piper\train-01.parquet
```

The service initializes the Persian resources at startup so missing resources fail
before it starts listening. Hugging Face model loading is configured offline;
the model/tokenizer files must already exist locally. These dependencies and data
belong to the service environment. The service does not download voices or install
packages on behalf of a client.

## RPC contract

The checked-in messages are generated from
[`sonata_grpc.proto`](../../source/synthDrivers/_sonata/sonata_grpc.proto), with
the original Sonata MIT license beside the generated module. The generic gRPC
handlers use the upstream service name `sonata_grpc.sonata_grpc` and method names.

* `GetSonataVersion` identifies this implementation as `sonata-piper-service/0.1.0`.
* `LoadVoice` accepts `default` for the first configured voice, a configured model
  stem such as `fa_IR-mana-medium`, its `.onnx` or `.onnx.json` filename, or the
  absolute model/configuration path supplied at startup. Other paths are rejected;
  an RPC cannot cause a new model or file to be loaded.
* Every `LoadVoice` returns a unique handle. Use that handle for `GetVoiceInfo`,
  `GetSynthesisOptions`, `SetSynthesisOptions`, and synthesis. Models are shared,
  while synthesis settings are private to each handle. Speaker options contain
  the **speaker name** from `VoiceInfo.speakers`, following upstream Sonata.
* `SynthesizeUtterance` supports unspecified or lazy synthesis mode. It returns
  raw mono signed PCM16 little-endian samples in messages of at most 64 KiB.
  `sample_width` is `2` bytes. `wav_samples` does not include a WAV file header.
  `rtf` is left at its protobuf default, `0`; no timing measurement is reported.
* The optional wire `rate` has Sonata's range `0..100`, corresponding to speed
  `0.5 + rate / 20`: `10` is normal speed. Omitting it preserves the configured
  length scale. Volume is `0..100`, with `100` as the default. Piper applies speed
  through its inference length scale, so output need not match Sonata's audio
  postprocessing. Pitch adjustment is unavailable; only omitted or neutral `50`
  is accepted. Appended silence is limited to 60 seconds.
* `supports_streaming_output` is false, and `SynthesizeUtteranceRealtime` returns
  `UNIMPLEMENTED`. Sending completed audio in bounded network messages does not
  make Piper inference generate its first audio earlier. The custom Persian
  frontend can process an entire input as one synthesis unit.

Text is limited to 256 KiB of UTF-8 per request. The service serializes inference
for each resident model. Cancelling an RPC discards subsequent audio and prevents
a waiting request from starting inference. An ONNX call already running cannot
be forcibly interrupted; the service checks cancellation before and after model
work and passes cancellation through the custom wheel's callback.

The protocol has no unload RPC. The service keeps at most 1024 voice handles and,
when full, reclaims handles unused for 24 hours. If all handles are recent,
`LoadVoice` returns `RESOURCE_EXHAUSTED`; restarting the service clears handles.
After restart, clients must call `LoadVoice` again.

The initial CLI binds IPv4 loopback only, without authentication or TLS. It is
intended for clients on the same computer; remote hosting is outside this version's
scope. Failed synthesis returns a short gRPC error and records diagnostic details
in the service log.

## Tests

With the service package installed, run from this directory:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

These tests use a real loopback gRPC server and a deterministic fake synthesis
engine. They cover the wire contract, session isolation, cancellation, bounded PCM,
input limits, and errors without requiring a voice model. Real inference and
Persian pronunciation also require a runtime check with the local wheel and assets.

See the [NVDA setup guide](../../SONATA_SERVICE_GUIDE.md) for configuring the client.
