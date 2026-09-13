# Sonata speech service integration

This fork includes a built-in Sonata synthesizer driver. It sends synthesis requests
to a separately running service using the upstream Sonata gRPC contract. NVDA receives
the audio and plays it through its selected output device. No NVDA add-on is required.

The included Python service runs the custom Piper wheel outside NVDA. Its inference
engine and Persian dependencies belong in a separate virtual environment; NVDA needs
only the gRPC client dependencies. The custom wheel, voice models, and Persian model
and dictionary assets are supplied separately and are not part of this branch.

## Install and start the Python service

From the repository root, create an independent service environment. These PowerShell
commands assume Python 3.13 is installed and the custom wheel is stored outside the
repository at the stated path:

```powershell
py -3.13 -m venv extras/piperService/.venv
.\extras\piperService\.venv\Scripts\python.exe -m pip install './extras/piperService[persian]' 'C:\Piper\piper_tts-1.3.1-cp39-abi3-win_amd64.whl'
```

Replace the wheel argument with its local path if stored elsewhere. A stock Piper
wheel does not provide the custom Persian phonemizer and cancellation extensions.
The `[persian]` extra installs the additional Persian dependencies. For voices that
do not use that phonemizer, install `./extras/piperService` without the extra.

Start with an existing model and its matching `.onnx.json` file:

```powershell
.\extras\piperService\.venv\Scripts\python.exe -m sonata_piper --voice 'C:\Piper\voices\fa_IR-mana-medium.onnx'
```

The service listens on `127.0.0.1:50051`. Repeat `--voice` to register more models;
`--port` changes the port. Keep the service running while using the driver.
NVDA does not start or install it automatically.

For enhanced Persian synthesis, provide both local assets explicitly:

```powershell
.\extras\piperService\.venv\Scripts\python.exe -m sonata_piper --voice 'C:\Piper\voices\fa_IR-mana-medium.onnx' --persian-phonemizer --ezafe-model 'C:\Piper\ezafe_model_quantized' --homograph-dictionary 'C:\Piper\train-01.parquet'
```

The service requires the Ezafe model directory and homograph dictionary when this
mode is enabled. Prepare the model's required tokenizer and model files in advance.
The service loads these resources at startup and enables Hugging Face and
Transformers offline mode. The custom wheel can have other resource requirements;
provision those locally too and verify synthesis with network access disabled.

See [the service README](extras/piperService/README.md) for its complete command-line
reference and standalone validation instructions.

## Configure NVDA

Set up and rebuild NVDA using its [development environment instructions](projectDocs/dev/createDevEnvironment.md)
and [build instructions](projectDocs/dev/buildingNVDA.md). A normal dependency sync
installs the Sonata client's `grpcio` and `protobuf` dependencies without the Piper
inference stack.

Start the service before selecting **Sonata speech service** in NVDA's synthesizer dialog. The
default configuration connects to `127.0.0.1:50051` and loads the voice named
`default`, which the Python service maps to the first `--voice` argument.

For another loopback endpoint or set of voices, edit the existing `speech` section in
your active `nvda.ini` while NVDA is stopped:

```ini
[speech]
    [[sonata]]
        endpoint = 127.0.0.1:50051
        voicePaths = default,
        requestTimeout = 30
```

`voicePaths` is a list; keep the trailing comma for a single entry. To load two
configured models, for example, use `voicePaths = fa_IR-mana-medium, en_US-lessac-medium`.
`requestTimeout` is measured in seconds and must be between 1 and 300. It limits
each synthesis request, including the time spent receiving its audio. Initial voice
loading has a separate two-second deadline so an unavailable service does not hold
up synthesizer selection. The Python service loads its models before listening.

The initial driver has no graphical endpoint editor; these connection settings are
configured in `nvda.ini` or, for the endpoint and voices, through the environment
variables below.

For a development run, environment variables can override the endpoint and voice
list without editing the configuration:

```powershell
$env:NVDA_SONATA_ENDPOINT = '127.0.0.1:50051'
$env:NVDA_SONATA_VOICES = '["default"]'
.\runnvda.bat
```

`NVDA_SONATA_VOICES` must contain a JSON array of voice identifiers or paths. The
Python service accepts `default`, configured model stems and filenames, and the
absolute model/configuration paths registered with `--voice`. It cannot load an
arbitrary unregistered path requested by a client. An upstream Sonata server uses
its own loading rules: supply explicit voice configuration paths on that server
instead of relying on the Python service's `default` alias.

The bundled service implements Sonata's public gRPC contract with the custom Python
Piper engine. It does not run the upstream Rust inference engine or require the
Sonata NVDA add-on. The unmodified wire definitions come from
[Sonata revision 451f9ebf](https://github.com/mush42/sonata/blob/451f9ebf2bd2aa2ba1be25fcec3b7593eeabf6ee/crates/frontends/grpc/proto/sonata_grpc.proto);
their MIT notice is retained next to the protocol source. Compatibility with another
Sonata server must be checked against that server's protocol version and voice
loading rules.

## Speech behavior and limits

The driver exposes Voice, Variant (speaker), Rate, and Volume. Rate 50 uses the
voice's normal speed; 0 and 100 request half and double speed. The bundled service
changes Piper's inference length scale; upstream Sonata can process speed changes
differently, so the resulting audio need not be identical. Enhanced Persian
processing is configured on the service command line.

* NVDA retains playback, queue ordering, pause/resume, cancellation, and speech
  progress notifications. Voice and synthesis settings accompany each request.
* Cancellation stops local playback and invalidates late audio. Whether inference
  itself stops immediately depends on the model and service.
* A request's deadline continues during a pause. If a long pause exceeds that
  deadline while audio is still being received, the unfinished request can expire.
  Increase `requestTimeout` for slower synthesis or longer pauses, within its limit.
* Audio arrives as a stream of chunks. This does not guarantee synthesis can yield
  audio partway through a sentence; time to first audio depends on the model's
  chunking and inference speed.
* This initial driver accepts loopback endpoints only. Remote computers, public
  hosting, authentication, and encrypted transport are outside this implementation.
* The driver is unavailable on the Windows secure desktop. NVDA uses its normal
  synthesizer fallback behavior there; do not depend on this service for sign-in
  or elevation prompts.

Running synthesis separately isolates its Python dependencies and process failures
from NVDA, while adding service startup, serialization, and communication costs.
Measure responsiveness with the intended voice and hardware.

## Validation and contribution scope

Run the focused driver tests in a configured NVDA development environment:

```powershell
.\rununittests.bat -k test_sonata
```

Run the service tests described in its README as well. For a runtime check, start
the service, select Sonata speech service, speak short and long text, interrupt with Control, and
pause/resume with Shift. Check configured voices, output device selection, and
recovery after stopping and restarting the service. Test Persian mode with its
local assets separately. Mocked tests do not establish real audio quality or latency.

This development branch is based on NVDA's `master` branch and contains the driver,
protocol definitions, standalone service source, and tests. Keep custom wheel, voice,
and training-data assets outside source changes proposed upstream. Upstream inclusion
still requires review of the dependency and packaging changes, behavior with real
audio, and NVDA's contribution requirements.
