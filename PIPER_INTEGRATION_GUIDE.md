# Piper synthesizer integration

This fork adds a Piper synthesizer with support for the custom Persian phonemizer.
Use NVDA's [development environment instructions](projectDocs/dev/createDevEnvironment.md)
and [build instructions](projectDocs/dev/buildingNVDA.md) for the required Python,
Visual Studio, submodules, and build commands.
Rebuild NVDA after updating its source or native submodules.

## Dependencies and local voices

The dependency manifest uses the custom
`piper_tts-1.3.1-cp39-abi3-win_amd64.whl` in the repository root.
This wheel extends Piper with `use_persian_phonemizer`, `ezafe_model_path`,
and `cancelled_callback`; a stock PyPI wheel is not a drop-in replacement.
Its additional Persian dependencies are declared and pinned in `pyproject.toml`
because the wheel does not declare them itself.
Install dependencies through NVDA's `uv` environment and keep `uv.lock` in sync.

Place each voice's `.onnx` model next to its matching `.onnx.json` configuration
in `source/synthDrivers/piper_voices`, or set `PIPER_VOICE_DIR` to another directory.
For example, `fa_IR-amir-medium.onnx` requires `fa_IR-amir-medium.onnx.json`.
The driver reads these files locally and does not download models or create directories.
It appears in NVDA's synthesizer list when the Piper package and a model/configuration pair exist.
Invalid voice metadata is skipped; initialization fails if no voice can be loaded.

For enhanced Persian synthesis, configure the Ezafe model path before starting NVDA:

```powershell
$env:PIPER_VOICE_DIR = 'C:\Piper\voices'
$env:PIPER_EZAFE_MODEL_PATH = 'C:\Piper\ezafe_model_quantized'
.\runnvda.bat
```

Select **Piper Neural TTS** in NVDA's synthesizer dialog.
Voice models, training data, and the custom wheel are local development assets;
NVDA's source-contribution checks impose a size limit on added files.
Keep those assets separate from source changes proposed upstream.

## Settings and speech behavior

- **Voice** selects a local model. The initial selection prefers NVDA's language.
- **Variant** selects a speaker within that model. Speaker choices refresh when the voice changes.
- **Rate** uses the model's normal speed at 50, half speed at 0, and double speed at 100.
- **Volume** adjusts synthesized audio from 0 to 100.
- **Use enhanced Persian phonemizer** defaults to disabled and is saved with the voice settings.
  Enable it after configuring the Ezafe model; existing saved preferences are restored normally.

The driver supports speech indices, timed breaks, inline rate and volume changes,
cancellation, and pause/resume. It preserves the voice, speaker, and settings associated
with queued speech. Completion is reported after pending audio drains.

Piper's synthesis configuration does not provide pitch control.
The driver does not advertise pitch, character-mode, or automatic language-switching
commands, so NVDA can apply its normal behavior for unsupported capabilities.

## Implementation

- `source/synthDrivers/piper.py` handles NVDA settings, voice discovery, and speech commands.
- `source/synthDrivers/_piper.py` performs synthesis and audio playback on a worker thread.

Cancellation invalidates active and pending utterances and interrupts the audio player.
Blocking audio feeds and drains do not hold the worker's state lock.
Pausing preserves queued speech; termination cancels speech and joins the worker before
releasing voices and unregistering the driver's settings callbacks.

## Validation

In a configured and rebuilt NVDA environment, run the focused regression tests:

```powershell
.\rununittests.bat -k test_piper
```

The tests mock inference and audio playback and cover cancellation races, pause/resume,
notification order, timed breaks, inline settings, and voice metadata.
Also run NVDA's normal lint checks and `uv lock --check`.

For a runtime check, select a local voice, read a long passage, interrupt it with Control,
and pause/resume it with Shift. Check voice/speaker changes, inline rate and volume,
and Persian synthesis with the configured Ezafe model.
