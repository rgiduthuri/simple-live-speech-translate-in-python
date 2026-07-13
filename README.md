# Live Speech-to-Speech Translation

**You speak English. It speaks Telugu back (and a few other Indic languagues) — live, and entirely on-device.**

```
mic → Whisper → IndicTrans2 → TTS → speakers
```

Every stage runs locally on the Apple Silicon GPU: no cloud, no API keys, no
audio ever leaves the machine. Measured on an M4 Pro, the Telugu voice starts
**~1.7 s** after you finish an English sentence. Any of AI4Bharat's 22 Indian
languages works, not just Telugu.

```sh
.venv/bin/python live_transcribe.py --language en --translate te --speak
```

The four stages are independently useful, so the app is usable at three levels.

**Transcription** — mic → Whisper → console. Runs OpenAI's open-source
**Whisper large-v3-turbo** on the GPU via
[mlx-whisper](https://pypi.org/project/mlx-whisper/), with Silero VAD
segmenting the mic stream into utterances. While you speak, a partial
transcript updates in place about once a second; when you pause (~0.4 s), the
line is finalized with a timestamp — final text lands ~1 s after you stop
talking. Falls back to
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CPU) if MLX is
unavailable. AI4Bharat's **IndicConformer-600M-multilingual** transcribes the
22 Indian languages directly:

```sh
.venv/bin/python live_transcribe.py                                    # English
.venv/bin/python live_transcribe.py --model indic-conformer --language te
```

**+ Translation** — each finalized English line is followed by its translation,
via AI4Bharat's **IndicTrans2**:

```sh
.venv/bin/python live_transcribe.py --language en --translate te
```

**+ Speech** — `--speak` synthesizes the translation with Meta's **MMS-TTS**
and plays it through the speakers, completing the loop.

While the TTS is audibly playing, the mic is ignored so the app doesn't
transcribe its own voice — anything you say during playback is dropped (a
log line reports each gated window). For gap-free continuous dictation use
headphones with `--tts-keep-mic`, or the fast `mms` TTS which keeps the
playback windows short.

## Requirements

- **Apple Silicon Mac** (M1 or newer). The GPU path uses Metal via MLX, which
  has no Intel build; an Intel Mac falls back to the slow CPU backend.
- **macOS 14 (Sonoma) or newer** — the floor set by the `torch` and `mlx` wheels.
- **~10 GB free disk**: ~1.6 GB of Python packages, plus 1.6–6.5 GB of model
  weights depending on which features you use.

To check: Apple menu → About This Mac shows the chip and macOS version.

## Setup

Written for someone who has never set up a developer environment on a Mac.
Each step is run once. If you already have Homebrew and `uv`, skip to step 4.

### 1. Open Terminal

Press `⌘-Space`, type `Terminal`, press Return. Everything below is typed into
this window, one command per line, each followed by Return. Commands that
change your system will ask for your login password — it stays invisible as you
type, which is normal.

### 2. Install the Xcode Command Line Tools

These provide the C compiler and `git`. Homebrew and several Python packages
need them.

```sh
xcode-select --install
```

A dialog appears — click **Install** and wait (a few minutes; it's a ~1 GB
download). If it says *"command line tools are already installed"*, you're done.
Verify:

```sh
xcode-select -p          # prints a path, e.g. /Library/Developer/CommandLineTools
```

### 3. Install Homebrew

[Homebrew](https://brew.sh) is the package manager used to install `uv`.

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

On Apple Silicon, Homebrew installs to `/opt/homebrew`, which is **not** on
your `PATH` by default — so the `brew` command won't be found until you add it.
The installer prints the two commands to run; they are:

```sh
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Verify:

```sh
brew --version           # prints e.g. Homebrew 4.x.x
```

### 4. Install uv (which supplies Python 3.12)

```sh
brew install uv
```

This project needs **Python 3.12**. Do not use the `python3` that ships with
macOS — it is 3.9 and is too old. You do not need to install Python separately:
`uv` downloads and manages a private CPython 3.12 for you in the next step.

### 5. Create the virtual environment and install dependencies

A virtual environment is a private folder of Python packages belonging to this
project alone, so it can't collide with anything else on your machine. Here it
lives in `.venv/`.

From inside the project folder (`cd` to wherever you cloned it):

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

The install pulls ~1.6 GB (PyTorch, MLX, and friends) and takes a few minutes.
Verify:

```sh
.venv/bin/python --version     # Python 3.12.x
```

You never need to "activate" the environment: every command below calls
`.venv/bin/python` directly, which uses it automatically.

### 6. Hugging Face account, token, and model permissions

The models are downloaded from [Hugging Face](https://huggingface.co). The
AI4Bharat ones are **gated**: you must be logged in, and your account must
accept each model's terms once. Acceptance is automatic and instant — it's a
click-through, not a review — but it is required, and skipping it produces a
`401` error on first run.

1. Create a free account at [huggingface.co/join](https://huggingface.co/join).
2. Create an access token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) —
   "New token" → type **Read** → copy it.
3. Log in (the `hf` command is installed inside the venv). Paste the token when
   prompted; it stays invisible as you type:

   ```sh
   .venv/bin/hf auth login
   ```

   Verify with `.venv/bin/hf auth whoami`, which prints your username.

4. Visit each gated model page **while logged in to the website** and click
   **Agree and access repository**. You only need the ones for the features you
   plan to use:

   | Model | Needed for |
   |---|---|
   | [indictrans2-en-indic-dist-200M](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M) | `--translate` (the default translator) |
   | [indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts) | `--tts-model parler` |
   | [indictrans2-en-indic-1B](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B) | `--translate-model 1B` |
   | [indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | `--model indic-conformer` |

   Plain English transcription needs no gated model, so you can skip this step
   entirely if you only want mic → Whisper → console.

### 7. First run and microphone permission

```sh
.venv/bin/python live_transcribe.py
```

The first run downloads Whisper (~1.6 GB) into `~/.cache/huggingface`; later
runs start in about a second. Adding `--translate te` or `--speak` downloads
their models on first use too (up to ~6.5 GB in total for full
speech-to-speech).

macOS will ask **"Terminal would like to access the microphone"** — click
**OK**. If you miss the prompt, the app appears to run but never transcribes
anything, because macOS silently hands it a stream of digital silence. Fix it
in **System Settings → Privacy & Security → Microphone** by switching on your
terminal app, then restart the terminal.

Now start speaking — a partial transcript appears while you talk, and each line
is finalized when you pause.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `command not found: brew` | Homebrew isn't on your `PATH`. Re-run the two `shellenv` lines in step 3, or open a new Terminal window. |
| `401 Client Error` / `access to <repo> is gated` | You're not logged in, or haven't accepted that model's terms. Redo step 6 — and note the terms must be accepted **per model**. |
| `No such file or directory: .venv/bin/python` | The virtual environment wasn't created, or you're in the wrong folder. `cd` to the project directory and re-run step 5. |
| App runs but transcribes nothing | Microphone permission (step 7), or your input volume is very low. Check **System Settings → Sound → Input** and speak — the level meter should move. |
| `uv: command not found` | Step 4 didn't complete. Run `brew install uv`. |
| Very slow transcription | You're on the CPU backend. Confirm the chip is Apple Silicon (`uname -m` prints `arm64`). |

## Run

```sh
.venv/bin/python live_transcribe.py
```

Ctrl-C flushes any pending audio and exits. Final transcripts are also
appended to `log/output_stt_en.txt` (see `--output-file` / `--no-output-file`).

## Options

| Flag | Purpose |
|------|---------|
| `--model small` | lower latency, less accurate |
| `--model indic-conformer --language te` | IndicConformer for Indian languages (requires `--language`) |
| `--decoder ctc` | IndicConformer decoding head (default `rnnt`) |
| `--translate [LANG]` | translate English finals with IndicTrans2 (bare flag = `te`); also appends to `log/output_stt_<LANG>.txt` |
| `--translate-model 1B` | larger IndicTrans2 variant (default `dist-200M`) |
| `--translate-beams 4` | translation beam size (default 1: ~3.5× faster, near-identical output) |
| `--speak` | speak the translation through the speakers (also in `--wav` replay) |
| `--tts-out [PATH]` | write TTS audio to a WAV instead of playing it (testing) |
| `--tts-model parler` | TTS engine: `mms` (fast, auto-matches `--translate` language), `parler` (higher quality; fp16 on the GPU at ~1× realtime, generated clause-by-clause so playback is gap-free — voice starts after the first clause is synthesized, ~3–4 s), or a HF repo |
| `--tts-voice Lalitha` | parler only: speaker name or full voice description (default: recommended speaker for the language) |
| `--list-models` | show known names for `--tts-model`, `--tts-voice`, `--translate-model` |
| `--tts-keep-mic` | don't gate the mic during TTS playback (headphones) |
| `--output-file PATH` | where final transcripts are appended (default `log/output_stt_en.txt`) |
| `--no-output-file` | console output only, no transcript file |
| `--save-audio [PATH]` | also save the captured mic audio as a WAV (default `log/capture_<timestamp>.wav`), plus a matching `.txt` transcript next to it |
| `--language en` | skip auto language detection |
| `--partial-interval 0` | disable in-progress partial updates |
| `--min-silence 0.7` | wait longer before ending an utterance |
| `--backend faster-whisper` | force the CPU (CTranslate2) backend |
| `--list-devices` / `--input-device N` | pick a microphone |
| `--wav file.wav` | run the pipeline on a 16 kHz mono WAV (testing) |

## Models

Every model is open-source and runs locally. Sizes are the on-disk download;
all are cached under `~/.cache/huggingface` on first use. "Gated" repos need a
one-click terms acceptance (see [step 6](#6-hugging-face-account-token-and-model-permissions)).

### Speech-to-text (`--model`, `--backend`)

| Model | Flag | Size | License | Gated |
|---|---|---|---|---|
| [whisper-large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) **(default)** | `--model large-v3-turbo` | 1.6 GB | MIT | no |
| [whisper-large-v3](https://huggingface.co/mlx-community/whisper-large-v3-mlx) | `--model large-v3` | 3.1 GB | MIT | no |
| [distil-whisper-large-v3](https://huggingface.co/mlx-community/distil-whisper-large-v3) | `--model distil-large-v3` | 1.5 GB | MIT | no |
| [whisper-medium](https://huggingface.co/mlx-community/whisper-medium-mlx) / [small](https://huggingface.co/mlx-community/whisper-small-mlx) / [base](https://huggingface.co/mlx-community/whisper-base-mlx) / [tiny](https://huggingface.co/mlx-community/whisper-tiny) | `--model medium` … `tiny` | 1.5 GB … 75 MB | MIT | no |
| [IndicConformer-600M](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | `--model indic-conformer` | 2.4 GB | MIT | **yes** |

Whisper transcribes ~99 languages (`--language`). IndicConformer transcribes
the 22 Indian languages below and requires `--language`. The CPU fallback
(`--backend faster-whisper`) uses the equivalent [Systran](https://huggingface.co/Systran)
CTranslate2 conversions.

### Translation (`--translate-model`)

Both are **English → Indic** only; the 22 languages below are targets.

| Model | Flag | Size | Speed | License | Gated |
|---|---|---|---|---|---|
| [IndicTrans2 dist-200M](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M) **(default)** | `--translate-model dist-200M` | 1.0 GB | 0.14 s/sentence | MIT | **yes** |
| [IndicTrans2 1B](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B) | `--translate-model 1B` | 4.2 GB | 0.35 s/sentence | MIT | **yes** |

### Text-to-speech (`--tts-model`)

| Model | Flag | Size | Speed | License | Gated |
|---|---|---|---|---|---|
| [MMS-TTS](https://huggingface.co/facebook/mms-tts-tel) **(default)** | `--tts-model mms` | ~140 MB per language | ~17× realtime, voice starts ~0.3 s | CC-BY-**NC** 4.0 | no |
| [indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts) | `--tts-model parler` | 3.5 GB | ~1× realtime, voice starts ~3–4 s | Apache-2.0 | **yes** |

MMS is the only realtime option and is the default. Parler is higher quality
and offers named speakers (`--tts-voice`); it also pulls in
[flan-t5-large](https://huggingface.co/google/flan-t5-large)'s tokenizer (3 MB,
Apache-2.0) for its voice-description encoder.

Silero VAD ships inside the faster-whisper wheel — no download.

## Supported languages

All 22 languages of the Indian constitution's Eighth Schedule can be
transcribed (IndicConformer) and translated into (IndicTrans2). **Speech output
is narrower**: 13 have an MMS voice, 11 have a parler voice, and **9 have no TTS
voice at all** — those can be transcribed and translated, but `--speak` will
refuse them.

| Code | Language | FLORES | Translate | MMS voice | Parler voice |
|---|---|---|---|---|---|
| `as` | Assamese | `asm_Beng` | ✅ | ✅ | ✅ |
| `bn` | Bengali | `ben_Beng` | ✅ | ✅ | ✅ |
| `brx` | Bodo | `brx_Deva` | ✅ | — | — |
| `doi` | Dogri | `doi_Deva` | ✅ | — | — |
| `gu` | Gujarati | `guj_Gujr` | ✅ | ✅ | ✅ |
| `hi` | Hindi | `hin_Deva` | ✅ | ✅ | ✅ |
| `kn` | Kannada | `kan_Knda` | ✅ | ✅ | ✅ |
| `kok` | Konkani | `gom_Deva` | ✅ | — | — |
| `ks` | Kashmiri | `kas_Arab` | ✅ | — | — |
| `mai` | Maithili | `mai_Deva` | ✅ | ✅ | — |
| `ml` | Malayalam | `mal_Mlym` | ✅ | ✅ | ✅ |
| `mni` | Manipuri (Meitei) | `mni_Beng` | ✅ | — | — |
| `mr` | Marathi | `mar_Deva` | ✅ | ✅ | ✅ |
| `ne` | Nepali | `npi_Deva` | ✅ | — | — |
| `or` | Odia | `ory_Orya` | ✅ | ✅ | ✅ |
| `pa` | Punjabi | `pan_Guru` | ✅ | ✅ | ✅ |
| `sa` | Sanskrit | `san_Deva` | ✅ | — | — |
| `sat` | Santali | `sat_Olck` | ✅ | — | — |
| `sd` | Sindhi | `snd_Arab` | ✅ | — | — |
| `ta` | Tamil | `tam_Taml` | ✅ | ✅ | ✅ |
| `te` | Telugu | `tel_Telu` | ✅ | ✅ | ✅ |
| `ur` | Urdu | `urd_Arab` | ✅ | ✅ | — |

`--translate` accepts either the short code or the FLORES code
(`--translate te` and `--translate tel_Telu` are equivalent). The **source**
language for translation is always English; the 22 above are targets.

### Parler speakers

`--tts-voice` takes one of these names (the first is the default for that
language) or a full voice description. Only for `--tts-model parler`.

| Language | Speakers |
|---|---|
| Assamese | Amit, Sita, Poonam, Rakesh |
| Bengali | Arjun, Aditi, Tapan, Rashmi, Arnav, Riya |
| Gujarati | Yash, Neha |
| Hindi | Rohit, Divya, Aman, Rani |
| Kannada | Suresh, Anu, Chetan, Vidya |
| Malayalam | Anjali, Harish, Anju |
| Marathi | Sanjay, Sunita, Nikhil, Radha, Varun, Isha |
| Odia | Manas, Debjani |
| Punjabi | Divjot, Gurpreet |
| Tamil | Jaya, Kavitha |
| Telugu | Prakash, Lalitha, Kiran |

`--list-models` prints all of the above from the running code.

## Design

See [DESIGN.md](DESIGN.md) for block diagrams of the pipeline,
threading model, and measured per-stage latencies.

## Authors

- Radhakrishna Giduthuri
- Fable 5

## Contribute
***Jump in and contribute whatever you’d like! Whether you want to build new features, speed things up, or squash some bugs.***

## Notes

- Decode cost is nearly flat in utterance length (Whisper pads to a 30 s
  window): ~0.95 s on the M4 Pro GPU vs ~2.6 s on CPU — hence the MLX default.
- For the CPU backend, `--compute-type float32` is the default: on Apple
  Silicon it runs via the Accelerate framework and measured ~1.7× faster
  than int8.
- Whisper models are multilingual; for English-only use,
  `--model distil-large-v3` is a fast, accurate alternative.
- Telugu TTS options, best quality first:
  [IndicF5](https://huggingface.co/ai4bharat/IndicF5) (MIT, voice-cloning,
  the top-rated open model in AI4Bharat's 2026 pairwise human evaluation,
  but seconds-per-sentence and a heavy dependency stack — not integrated),
  `--tts-model parler` =
  [indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts)
  (Apache-2.0, ~3 s/sentence on the M4 GPU), and the default
  `--tts-model mms` = [MMS-TTS](https://huggingface.co/facebook/mms-tts-tel)
  (CC-BY-**NC** 4.0, near-instant, the only realtime option).
