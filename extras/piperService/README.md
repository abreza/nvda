# Standalone Piper service

This service runs the custom Piper engine in its own Python environment and exposes
the Sonata gRPC interface on `127.0.0.1:50051`. The built-in NVDA Sonata driver and
other gRPC clients can use it without importing Piper into NVDA. It is a Python
implementation of Sonata's RPC interface; it does not run Sonata's Rust engine.

## Install and run

The initial backend requires the project's custom Windows x64 Piper 1.3.1 wheel.
Its `PiperVoice` API adds `use_persian_phonemizer`, `ezafe_model_path`,
`use_short_speech_repeat`, and `cancelled_callback`. A stock PyPI Piper wheel is not
a compatible replacement.
Keep the wheel and voice assets outside the NVDA source checkout. With Python 3.13
installed, run these commands from `extras/piperService`:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\python.exe -m pip install "C:\Piper\piper_tts-1.3.1-cp39-abi3-win_amd64.whl[alignment]"
.\.venv\Scripts\python.exe -m piper.prepare_short_speech `
    --voice C:\Piper\voices\fa_IR-mana-medium.onnx `
    --output C:\Piper\voices\fa_IR-mana-medium.short-repeat.onnx
.\.venv\Scripts\python.exe -m sonata_piper --voice C:\Piper\voices\fa_IR-mana-medium.short-repeat.onnx
```

The matching `.onnx.json` must sit beside the `.onnx` file. Repeat `--voice` to
preload additional voices; use `--port` to change the loopback port. The service
loads configured models once before accepting connections and keeps them resident.
Start it independently of NVDA and leave the process running while clients use it.
Stop with Control+C.

This NVDA service enables short Persian word repetition by default and requires a
prepared model. Standalone Piper keeps repetition disabled by default.
Preparation creates a separate model/configuration pair and preserves the original;
run it once for each output path. To use the original model with ordinary synthesis,
pass `--no-short-speech-repeat` to the service.

To enable the custom Persian phonemizer, install its extra dependencies and provide
the local Ezafe model and homograph dictionary explicitly:

```powershell
.\.venv\Scripts\python.exe -m pip install ".[persian]"
.\.venv\Scripts\python.exe -m sonata_piper `
    --voice C:\Piper\voices\fa_IR-mana-medium.short-repeat.onnx `
    --persian-phonemizer `
    --ezafe-model C:\Piper\ezafe_model_quantized `
    --homograph-dictionary C:\Piper\train-01.parquet
```

The service initializes the Persian resources at startup so missing resources fail
before it starts listening. Hugging Face model loading is configured offline;
the model/tokenizer files must already exist locally. These dependencies and data
belong to the service environment. The service does not download voices or install
packages on behalf of a client.

### Experimental Mana short speech

The updated custom Piper wheel provides a workaround for isolated Persian words:
it generates internal repetitions and returns only the first copy, cropped
using the model's predicted durations. All repetition, cropping, and audio
processing live in Piper; this service enables its option by default.

After rebuilding the custom wheel, replace the older version in the service
environment with `python -m pip install --force-reinstall --no-deps
C:\Piper\piper_tts-1.3.1-cp39-abi3-win_amd64.whl`.

Prepare a separate model using the install commands above. Preparation requires
`onnx` (the custom wheel's `alignment` extra). Then start the updated wheel with:

```powershell
.\.venv\Scripts\python.exe -m sonata_piper `
    --voice C:\Piper\voices\fa_IR-mana-medium.short-repeat.onnx `
    --persian-phonemizer `
    --ezafe-model C:\Piper\ezafe_model_quantized `
    --homograph-dictionary C:\Piper\train-01.parquet
```

Use `--no-short-speech-repeat` to disable the workaround; `--short-speech-repeat`
explicitly enables it. Listening confirmed that extracting `دستیار`
from a repetition improves that word; some isolated letters still fail. It is an
experimental workaround, adds inference work, and does not establish correct
pronunciation for all short input. The original voice file is preserved. NVDA
continues sending ordinary text; no driver-side repetition is needed. An older custom
wheel without the repetition option can be used with `--no-short-speech-repeat`.

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

The default tests use a real loopback gRPC server and a deterministic fake synthesis
engine. They cover the wire contract, session isolation, cancellation, bounded PCM,
input limits, and errors without requiring a voice model. Real inference and
Persian pronunciation also require a runtime check with the local wheel and assets.

The opt-in `tests/test_mana.py` checks the real engine and gRPC transport for
`دستیار` and its individual letters. Set `SONATA_TEST_VOICE` to the local model,
`SONATA_TEST_EZAFE_MODEL` to the Ezafe directory, and
`SONATA_TEST_HOMOGRAPH_DICTIONARY` to the parquet file, then run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_mana.py -v
```

It compares the engine's PCM with the received PCM byte for byte and prints time
to first audio. Without all three variables, this test is skipped.
Repetition is enabled by default, so select the prepared model. Set
`SONATA_TEST_SHORT_SPEECH_REPEAT=0` to use an original model with ordinary synthesis.

To compare the standard eSpeak frontend with the enhanced Persian frontend using
the same local voice, run this opt-in diagnostic in the service environment:

```powershell
.\.venv\Scripts\python.exe -m sonata_piper.diagnose `
    --voice C:\Piper\voices\fa_IR-mana-medium.short-repeat.onnx `
    --ezafe-model C:\Piper\ezafe_model_quantized `
    --homograph-dictionary C:\Piper\train-01.parquet `
    --output C:\Piper\diagnostics\mana
```

It writes a WAV per input/frontend and `report.json`. Default inputs include
`دستیار`, the word with a period, a sentence containing it, its individual letters,
expanded letter names, and two explicit IPA candidates. The candidates are listening
comparisons, not verified pronunciation targets. Repeat `--text` to replace the
default cases, for example `--text "س" --text "س."`; use `--hash-assets` to record
the voice/configuration SHA256 hashes.
The diagnostic enables repetition by default and records that choice in the report.
Pass `--no-short-speech-repeat` to compare ordinary synthesis or use an original model.

The report captures the phoneme chunks and IDs actually used by synthesis, missing
symbols, warnings (including fallback warnings), loaded module paths and versions,
frontend/inference timing, time to first PCM, audio duration, peak/RMS, and nonzero
sample counts. Leading/trailing samples below 1% of PCM full scale are also measured;
that threshold is an activity indicator, not a speech recognition result. The
command loads one voice and its Persian resources, then switches its frontend flag
for the two modes. Model loading is timed separately; the first case in each mode
may include lazy initialization costs.
The voice's default inference settings are retained; stochastic synthesis can vary
between runs even when the phonemes are identical.

`raw_inference_audio` records peak/RMS before Piper's automatic peak normalization.
`acoustics` measures the fraction of active-frame energy above 4 kHz, and
`acoustic_flags` reports `high_frequency_dominant` when its median exceeds 90%.
This can expose quiet high-frequency artifacts that normalization amplifies into
loud hiss. It is a heuristic: a high-frequency sound can also be intentional.
`acoustic_flags_detected` summarizes these flags independently of structural
`automated_checks_passed` and the exit status.

The command exits with status 1 for exceptions, missing phoneme symbols, or empty or
all-zero audio. Passing these checks does **not** establish correct pronunciation:
listen to the WAVs and compare their phonemes. It exercises the service backend;
NVDA's character announcements, cancellation, gRPC transport, and playback require
separate tests. It uses the same offline Persian resource loading as the service.

See the [NVDA setup guide](../../SONATA_SERVICE_GUIDE.md) for configuring the client.
