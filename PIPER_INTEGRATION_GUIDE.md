# NVDA Build & Piper TTS Integration Guide

## ⚠️ Important: Windows Only

**NVDA is a Windows-only screen reader.** You cannot build or run NVDA on macOS. You need a Windows 10/11 machine.

---

## Part 1: Prerequisites (Windows)

### 1.1 Install Python 3.13

Download and install Python 3.13.x (64-bit) from [python.org](https://www.python.org/downloads/).

During installation:
- ✅ Add Python to PATH
- ✅ Install pip

### 1.2 Install uv (Package Manager)

Open PowerShell as Administrator and run:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Or with pip:
```powershell
pip install uv
```

### 1.3 Install Visual Studio 2022

Download [Visual Studio 2022 Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe) or [Community Edition](https://aka.ms/vs/17/release/vs_Community.exe).

During installation, select:
- **Desktop development with C++**
  - Include "C++ Clang tools for Windows"
- Individual Components:
  - Windows 11 SDK (10.0.26100.x)
  - MSVC v143 - VS 2022 C++ ARM64/ARM64EC build tools
  - MSVC v143 - VS 2022 C++ x64/x86 build tools
  - C++ ATL for v143 build tools (x86 & x64)
  - C++ ATL for v143 build tools (ARM64/ARM64EC)

**Quick method:** Import the `.vsconfig` file from the NVDA repository:
```powershell
vs_installer.exe --config "C:\path\to\nvda-master\.vsconfig"
```

### 1.4 Install Git

Download from [git-scm.com](https://git-scm.com/download/win).

---

## Part 2: Clone NVDA Repository

```powershell
# Clone with submodules
git clone --recursive https://github.com/nvaccess/nvda.git
cd nvda

# Or if already cloned without --recursive:
git submodule update --init --recursive
```

---

## Part 3: Build NVDA

### 3.1 Build the Project

Open a **Developer Command Prompt for VS 2022** (or regular PowerShell) and navigate to the NVDA directory:

```powershell
cd C:\path\to\nvda-master

# Build NVDA (this creates the virtual environment and builds everything)
.\scons.bat
```

This will:
1. Create a Python virtual environment
2. Install all dependencies via uv
3. Build nvdaHelper (C++ components)
4. Compile all resources

### 3.2 Run NVDA from Source

```powershell
.\runnvda.bat
```

Or with debug options:
```powershell
.\runnvda.bat --debug-logging
```

---

## Part 4: Install Your Custom Piper TTS Package

### 4.1 Build the Piper Wheel (on your development machine)

First, build your custom Piper package with Persian phonemizer:

```powershell
# Navigate to your Piper project directory
cd C:\path\to\piper-project

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install build dependencies
pip install scikit-build setuptools wheel cmake ninja

# Build the wheel
python setup.py bdist_wheel
```

The wheel will be created in `dist/piper_tts-1.3.1-cp39-abi3-win_amd64.whl` (or similar).

### 4.2 Install Piper into NVDA's Virtual Environment

```powershell
# Navigate to NVDA directory
cd C:\path\to\nvda-master

# Activate NVDA's virtual environment
.\.venv\Scripts\Activate.ps1

# Install your custom Piper wheel
pip install C:\path\to\piper-project\dist\piper_tts-1.3.1-cp39-abi3-win_amd64.whl

# Install additional dependencies for Persian phonemizer
pip install hazm pandas pyarrow transformers optimum[onnxruntime]
```

### 4.3 Set Up Voice Directory

Create the voice directory and copy your Piper voices:

```powershell
# Create voice directory
mkdir "C:\path\to\nvda-master\source\synthDrivers\piper_voices"

# Copy your ONNX voice files
# Each voice needs: voice.onnx and voice.onnx.json
copy "C:\path\to\fa_IR-mana-medium.onnx" "C:\path\to\nvda-master\source\synthDrivers\piper_voices\"
copy "C:\path\to\fa_IR-mana-medium.onnx.json" "C:\path\to\nvda-master\source\synthDrivers\piper_voices\"
```

### 4.4 Set Environment Variables for Persian Phonemizer

Set these environment variables before running NVDA:

```powershell
# Set voice directory
$env:PIPER_VOICE_DIR = "C:\path\to\nvda-master\source\synthDrivers\piper_voices"

# Set Ezafe model path (for Persian phonemizer)
$env:PIPER_EZAFE_MODEL_PATH = "C:\path\to\ezafe_model_quantized"

# Set Homograph dictionary path
$env:HOMOGRAPH_DICT_PATH = "C:\path\to\homograph_dictionary.parquet"
```

Or add them permanently via System Properties → Environment Variables.

---

## Part 5: Configure NVDA to Use Piper

### 5.1 Run NVDA

```powershell
.\runnvda.bat
```

### 5.2 Select Piper as Synthesizer

1. Press **NVDA+Ctrl+S** to open Settings
2. Go to **Speech** category
3. Change **Synthesizer** to "Piper TTS"
4. Select your voice from the **Voice** dropdown
5. Click **OK**

---

## Part 6: Project Structure

Your Piper integration files in NVDA:

```
source/synthDrivers/
├── piperDriver.py      # Main Piper synth driver
├── _piper.py           # Background synthesis thread
└── piper_voices/       # Voice files directory
    ├── fa_IR-mana-medium.onnx
    ├── fa_IR-mana-medium.onnx.json
    └── ... (other voices)
```

---

## Part 7: Troubleshooting

### Issue: "Piper TTS package not installed"

```powershell
# Activate NVDA venv and reinstall
.\.venv\Scripts\Activate.ps1
pip install --force-reinstall C:\path\to\piper_tts-*.whl
```

### Issue: Persian phonemizer not working

Ensure the Ezafe model is properly installed:

```powershell
# Download and extract the Ezafe model
# Set the environment variable
$env:PIPER_EZAFE_MODEL_PATH = "C:\path\to\ezafe_model_quantized"
```

### Issue: No voices found

Check that:
1. Voice files are in the correct directory
2. Both `.onnx` and `.onnx.json` files exist
3. `PIPER_VOICE_DIR` environment variable is set correctly

### Issue: Build errors

```powershell
# Clean and rebuild
.\scons.bat -c
.\scons.bat
```

### View NVDA logs

```powershell
# Logs are in:
%TEMP%\nvda-*.log
# Or with debug logging:
.\runnvda.bat --debug-logging -f %TEMP%\nvda-debug.log
```

---

## Part 8: Creating an NVDA Installer with Piper

To create a distributable NVDA installer:

```powershell
# Build the launcher/installer
.\scons.bat launcher

# The installer will be in the output directory
dir .\output\
```

---

## Quick Start Commands Summary

```powershell
# 1. Build NVDA
cd C:\path\to\nvda-master
.\scons.bat

# 2. Install Piper
.\.venv\Scripts\Activate.ps1
pip install C:\path\to\piper_tts-*.whl
pip install hazm pandas pyarrow transformers optimum[onnxruntime]

# 3. Set environment variables
$env:PIPER_VOICE_DIR = "C:\path\to\nvda-master\source\synthDrivers\piper_voices"
$env:PIPER_EZAFE_MODEL_PATH = "C:\path\to\ezafe_model_quantized"

# 4. Run NVDA
.\runnvda.bat
```

---

## Additional Resources

- [NVDA Developer Guide](https://github.com/nvaccess/nvda/blob/master/projectDocs/dev/createDevEnvironment.md)
- [Piper TTS Documentation](https://github.com/rhasspy/piper)
- [NVDA Add-on Development](https://github.com/nvaccess/nvda/blob/master/projectDocs/dev/addonDevelopment.md)
