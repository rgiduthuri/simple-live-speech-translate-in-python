#!/usr/bin/env python3
"""Live microphone transcription: mic -> Whisper -> console.

Audio is captured at 16 kHz mono and segmented into utterances with Silero
VAD. While an utterance is in progress, the buffer is re-decoded about once
a second and shown as an updating partial line; when trailing silence ends
the utterance, the line is finalized with a timestamp.

Inference runs on the Apple Silicon GPU via mlx-whisper when available,
falling back to faster-whisper (CPU) otherwise.

Usage:
    python live_transcribe.py                    # live mic, large-v3-turbo
    python live_transcribe.py --model small      # faster, less accurate
    python live_transcribe.py --list-devices     # show input devices
    python live_transcribe.py --wav test.wav     # run pipeline on a file

Authors: Radhakrishna Giduthuri, Fable 5.
"""

import argparse
import os
import queue
import re
import shutil
import sys
import threading
import time
import wave
from datetime import datetime

import numpy as np

SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.1          # mic callback block size
VAD_CHECK_SECONDS = 0.1      # how often to re-run VAD on the buffer
NO_SPEECH_KEEP_SECONDS = 1.0 # rolling buffer kept while silent

INDIC_CONFORMER_REPO = "ai4bharat/indic-conformer-600m-multilingual"
INDIC_LANGUAGES = ("as bn brx doi gu hi kn kok ks mai ml mni mr ne or pa "
                   "sa sat sd ta te ur").split()

TRANSLATE_REPOS = {
    "dist-200M": "ai4bharat/indictrans2-en-indic-dist-200M",
    "1B": "ai4bharat/indictrans2-en-indic-1B",
}

# indic-parler-tts speakers per language, recommended ones first
# (from the model card).
PARLER_SPEAKERS = {
    "as": ["Amit", "Sita", "Poonam", "Rakesh"],
    "bn": ["Arjun", "Aditi", "Tapan", "Rashmi", "Arnav", "Riya"],
    "gu": ["Yash", "Neha"],
    "hi": ["Rohit", "Divya", "Aman", "Rani"],
    "kn": ["Suresh", "Anu", "Chetan", "Vidya"],
    "ml": ["Anjali", "Harish", "Anju"],
    "mr": ["Sanjay", "Sunita", "Nikhil", "Radha", "Varun", "Isha"],
    "or": ["Manas", "Debjani"],
    "pa": ["Divjot", "Gurpreet"],
    "ta": ["Jaya", "Kavitha"],
    "te": ["Prakash", "Lalitha", "Kiran"],
}
VOICE_TEMPLATE = ("{name} speaks with a clear, moderate-pitched voice at a "
                  "normal pace with very clear audio.")
# Short language codes -> FLORES codes used by IndicTrans2.
FLORES_CODES = {
    "as": "asm_Beng", "bn": "ben_Beng", "brx": "brx_Deva", "doi": "doi_Deva",
    "gu": "guj_Gujr", "hi": "hin_Deva", "kn": "kan_Knda", "kok": "gom_Deva",
    "ks": "kas_Arab", "mai": "mai_Deva", "ml": "mal_Mlym", "mni": "mni_Beng",
    "mr": "mar_Deva", "ne": "npi_Deva", "or": "ory_Orya", "pa": "pan_Guru",
    "sa": "san_Deva", "sat": "sat_Olck", "sd": "snd_Arab", "ta": "tam_Taml",
    "te": "tel_Telu", "ur": "urd_Arab",
}

# Short model names -> MLX community conversions on the HF hub.
MLX_REPOS = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "tiny": "mlx-community/whisper-tiny",
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
}


def log(text):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] * {text}", file=sys.stderr, flush=True)


class Translator:
    """English -> Indic translation with AI4Bharat IndicTrans2."""

    def __init__(self, target, model_key, beams=1):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor

        self.beams = beams
        self.tgt_lang = FLORES_CODES.get(target, target)
        if "_" not in self.tgt_lang:
            sys.exit("error: --translate must be a FLORES code or one of: "
                     + " ".join(FLORES_CODES))
        repo = TRANSLATE_REPOS[model_key]
        log(f"loading translator '{repo}' -> {self.tgt_lang}...")
        t0 = time.time()
        self._torch = torch
        self._ip = IndicProcessor(inference=True)
        self._tok = hf_from_pretrained(AutoTokenizer.from_pretrained,
                                       repo, trust_remote_code=True)
        self._model = hf_from_pretrained(
            AutoModelForSeq2SeqLM.from_pretrained, repo,
            trust_remote_code=True)
        self.translate("Warming up the translator.")
        log(f"translator ready in {time.time() - t0:.1f}s")

    def translate(self, text):
        """Translate text, returning a list of translated sentences."""
        # IndicTrans2 is trained on single sentences; split and batch.
        sents = [s for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s]
        if not sents:
            return []
        batch = self._ip.preprocess_batch(sents, src_lang="eng_Latn",
                                          tgt_lang=self.tgt_lang)
        inputs = self._tok(batch, truncation=True, padding="longest",
                           return_tensors="pt")
        with self._torch.no_grad():
            out = self._model.generate(**inputs, max_length=256,
                                       num_beams=self.beams)
        decoded = self._tok.batch_decode(out, skip_special_tokens=True)
        return self._ip.postprocess_batch(decoded, lang=self.tgt_lang)


# Verified against the HF hub 2026-07-12; the other 9 supported languages
# (brx doi kok ks mni ne sa sat sd) have no MMS voice at all.
MMS_TTS_REPOS = {
    "as": "facebook/mms-tts-asm", "bn": "facebook/mms-tts-ben",
    "gu": "facebook/mms-tts-guj", "hi": "facebook/mms-tts-hin",
    "kn": "facebook/mms-tts-kan", "mai": "facebook/mms-tts-mai",
    "ml": "facebook/mms-tts-mal", "mr": "facebook/mms-tts-mar",
    "or": "facebook/mms-tts-ory", "pa": "facebook/mms-tts-pan",
    "ta": "facebook/mms-tts-tam", "te": "facebook/mms-tts-tel",
    "ur": "facebook/mms-tts-urd-script_arabic",
}


def resolve_tts(name, translate_lang):
    """Map a --tts-model alias or repo to (hf_repo, engine)."""
    if name == "mms":
        repo = MMS_TTS_REPOS.get(translate_lang)
        if repo is None:
            sys.exit(f"error: MMS has no TTS voice for '{translate_lang}'; "
                     f"use --tts-model parler (or a HF repo). MMS covers: "
                     + " ".join(sorted(MMS_TTS_REPOS)))
        return repo, "vits"
    if name == "parler":
        return "ai4bharat/indic-parler-tts", "parler"
    return name, ("parler" if "parler" in name.lower() else "vits")


def hf_from_pretrained(loader, repo, **kwargs):
    """from_pretrained with an actionable message for gated repos."""
    try:
        return loader(repo, **kwargs)
    except OSError as e:
        if "gated" in str(e).lower() or "restricted" in str(e).lower():
            sys.exit(f"error: access to {repo} is gated. Log in with "
                     f"'hf auth login' and accept the terms at "
                     f"https://huggingface.co/{repo}")
        raise


class Speaker:
    """Text-to-speech with sequential, non-blocking playback.

    Engines: 'vits' (MMS et al., fast on CPU) and 'parler'
    (indic-parler-tts, higher quality, runs on the Apple GPU via MPS).

    In mic mode, audio is pushed into a ring buffer consumed by a
    persistent output stream. Parler runs in fp16 on MPS (~1.0x realtime;
    fp32 is only ~0.7x) and text is split into clauses: each clause is
    generated in full (no mid-clause underruns possible) while the
    previous one plays, so waits land on natural pause boundaries.
    is_busy() lets the transcriber gate the mic against feedback. In
    --wav test mode (wav_out set), audio is appended to a WAV file
    instead.
    """

    PREBUFFER_SECONDS = 1.0
    TAIL_SECONDS = 0.3
    MIN_CLAUSE_CHARS = 12
    MAX_CLAUSE_CHARS = 48
    MAX_CONSECUTIVE_FAILURES = 3
    BACKLOG_WARN_CLAUSES = 8
    # ~86 DAC frames per second of audio; budget generation per clause so
    # a missed EOS can't stall the worker for the model's full 30 s cap.
    PARLER_FRAMES_PER_SECOND = 86.0

    CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:।])\s+")

    def __init__(self, args, wav_out=None, warmup_text=None):
        import torch
        from transformers import AutoTokenizer

        repo, self.engine = resolve_tts(args.tts_model, args.translate)
        log(f"loading TTS '{repo}' ({self.engine})...")
        t0 = time.time()
        self._torch = torch
        if self.engine == "vits":
            from transformers import VitsModel
            self._device = "cpu"
            self._tok = AutoTokenizer.from_pretrained(repo)
            self._model = VitsModel.from_pretrained(repo)
        else:
            # parler-tts warns "Flash attention 2 is not installed" at
            # import; flash-attn is CUDA-only (impossible on macOS) and
            # eager attention is the fallback we run anyway.
            import logging
            logging.getLogger("parler_tts").setLevel(logging.ERROR)
            from parler_tts import ParlerTTSForConditionalGeneration
            self._device = ("mps" if torch.backends.mps.is_available()
                            else "cpu")
            dtype = (torch.float16 if self._device == "mps"
                     else torch.float32)
            # attn_implementation pinned to eager: flash-attn is
            # CUDA-only, so this is what parler falls back to anyway —
            # stating it silences the "Flash attention 2 is not
            # installed" warning.
            self._model = hf_from_pretrained(
                ParlerTTSForConditionalGeneration.from_pretrained, repo,
                torch_dtype=dtype, attn_implementation="eager").to(
                    self._device)
            self._tok = AutoTokenizer.from_pretrained(repo)
            desc_tok = AutoTokenizer.from_pretrained(
                self._model.config.text_encoder._name_or_path)
            voice = args.tts_voice
            if voice is None:
                speakers = PARLER_SPEAKERS.get(args.translate)
                voice = VOICE_TEMPLATE.format(
                    name=speakers[0] if speakers else "A speaker")
            elif " " not in voice:  # bare speaker name
                voice = VOICE_TEMPLATE.format(name=voice)
            log(f"TTS voice: {voice}")
            desc = desc_tok(voice, return_tensors="pt")
            self._desc = desc.to(self._device)
        self.sample_rate = self._model.config.sampling_rate
        # Gate only in live mic mode: in --wav replay the "mic" is the
        # file, and gating would discard the input we're transcribing.
        self.gate_mic = not args.tts_keep_mic and not args.wav
        # Warm up with a representative-length clause in the target
        # language so the first real utterance generates at full speed
        # (MMS tokenizers drop text in the wrong script, making a
        # wrong-language warmup a silent no-op).
        self.synthesize(warmup_text or "వాతావరణం చాలా బాగుంది")
        log(f"TTS ready in {time.time() - t0:.1f}s")

        # Unlocked reads of _pending/_gen_active/_buffered in is_busy()
        # and the playback callback rely on CPython GIL atomicity plus the
        # worker's write ordering (push -> gen_active off -> pending
        # decrement); verified safe on CPython 3.12. Writes are locked for
        # future-proofing — do not remove the lock as an optimization.
        self._lock = threading.Lock()
        self._chunks = []           # queued audio, list of float32 arrays
        self._buffered = 0          # total samples in _chunks
        self._playing = False
        self._gen_active = False
        self._last_busy = 0.0
        self._failures = 0          # consecutive synthesis failures
        self._last_audible = 0.0
        self.disabled = False
        self._wav_out = None
        if wav_out:
            os.makedirs(os.path.dirname(wav_out) or ".", exist_ok=True)
            self._wav_out = wave.open(wav_out, "wb")
            self._wav_out.setnchannels(1)
            self._wav_out.setsampwidth(2)
            self._wav_out.setframerate(self.sample_rate)
            log(f"writing TTS audio to {wav_out}")
        else:
            import sounddevice as sd
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate, channels=1, dtype="float32",
                callback=self._playback_callback)
            self._stream.start()
        self._queue = queue.Queue()
        self._pending = 0
        threading.Thread(target=self._worker, daemon=True).start()

    def synthesize(self, text):
        inputs = self._tok(text, return_tensors="pt")
        if inputs["input_ids"].shape[-1] == 0:
            return np.zeros(0, dtype=np.float32)
        with self._torch.no_grad():
            if self.engine == "vits":
                return self._model(**inputs).waveform[0].numpy()
            inputs = inputs.to(self._device)
            expected_seconds = 0.12 * len(text) + 3.0
            budget = min(2580, int(self.PARLER_FRAMES_PER_SECOND
                                   * expected_seconds))
            audio = self._model.generate(
                input_ids=self._desc.input_ids,
                attention_mask=self._desc.attention_mask,
                prompt_input_ids=inputs.input_ids,
                prompt_attention_mask=inputs.attention_mask,
                max_new_tokens=budget)
            audio = (audio.float().cpu().numpy().squeeze()
                     .astype(np.float32))
            if (len(audio) / self.sample_rate
                    >= budget / self.PARLER_FRAMES_PER_SECOND - 0.2):
                log("TTS clause hit its generation cap; audio may be "
                    "clipped")
            return audio

    def _clauses(self, text):
        """Split text into MIN..MAX_CLAUSE_CHARS pieces at punctuation,
        falling back to word boundaries for overlong clauses."""
        parts = [p for p in self.CLAUSE_SPLIT_RE.split(text.strip()) if p]
        clauses, current = [], ""
        for part in parts:
            current = f"{current} {part}".strip()
            if len(current) < self.MIN_CLAUSE_CHARS:
                continue
            while len(current) > self.MAX_CLAUSE_CHARS:
                cut = current.rfind(" ", self.MIN_CLAUSE_CHARS,
                                    self.MAX_CLAUSE_CHARS)
                if cut < 0:
                    break
                clauses.append(current[:cut])
                current = current[cut + 1:]
            clauses.append(current)
            current = ""
        if current:
            if clauses:
                clauses[-1] += " " + current
            else:
                clauses.append(current)
        return clauses

    def say(self, text):
        if self.disabled:
            return
        # Parler generates at ~1x realtime, so per-clause generation can
        # chain seamlessly behind playback; whole sentences could not.
        clauses = (self._clauses(text) if self.engine == "parler"
                   else [text])
        for clause in clauses:
            with self._lock:
                self._pending += 1
                if self._pending == self.BACKLOG_WARN_CLAUSES:
                    log(f"TTS backlog: {self._pending} clauses queued; "
                        f"speech output is lagging")
            self._queue.put(clause)

    def is_busy(self):
        """True while speech is queued, generating, or playing (+ tail)."""
        if self._pending > 0 or self._gen_active or self._buffered > 0:
            self._last_busy = time.monotonic()
            return True
        return time.monotonic() - self._last_busy < self.TAIL_SECONDS

    def is_audible(self):
        """True only while the speakers are (about to be) emitting sound.

        This is the mic-gate condition: unlike is_busy(), it stays False
        during parler's silent generation lead, so the app keeps
        listening until audio actually reaches the output path.
        """
        if self._buffered > 0 or self._playing:
            self._last_audible = time.monotonic()
            return True
        return time.monotonic() - self._last_audible < self.TAIL_SECONDS

    def _push(self, audio):
        with self._lock:
            self._chunks.append(audio)
            self._buffered += len(audio)

    def _playback_callback(self, outdata, frames, time_info, status):
        out = outdata[:, 0]
        out.fill(0.0)
        with self._lock:
            if not self._playing:
                # Jitter buffer: while generation is running or queued,
                # wait for PREBUFFER seconds; once all done, play the rest.
                # (_gen_active alone flickers False between queue items.)
                need = (int(self.PREBUFFER_SECONDS * self.sample_rate)
                        if self._gen_active or self._pending > 0 else 1)
                if self._buffered < need:
                    return
                self._playing = True
            filled = 0
            while filled < frames and self._chunks:
                chunk = self._chunks[0]
                n = min(len(chunk), frames - filled)
                out[filled:filled + n] = chunk[:n]
                if n == len(chunk):
                    self._chunks.pop(0)
                else:
                    self._chunks[0] = chunk[n:]
                filled += n
                self._buffered -= n
            if not self._chunks:
                self._playing = False  # underrun or done: rebuffer

    def _worker(self):
        while True:
            text = self._queue.get()
            self._gen_active = True
            try:
                audio = self.synthesize(text)
                if len(audio):
                    if self._wav_out:
                        pcm = (np.clip(audio, -1.0, 1.0)
                               * 32767).astype(np.int16)
                        self._wav_out.writeframes(pcm.tobytes())
                    else:
                        self._push(audio)
                self._failures = 0
            except Exception as e:
                # The worker must survive: if it dies, _pending never
                # reaches 0 and the feedback gate mutes the mic forever.
                self._failures += 1
                log(f"TTS synthesis failed "
                    f"({self._failures}/{self.MAX_CONSECUTIVE_FAILURES}): "
                    f"{e!r}")
                if (self._failures >= self.MAX_CONSECUTIVE_FAILURES
                        and not self.disabled):
                    self.disabled = True
                    log("TTS disabled after repeated failures; "
                        "transcription continues without speech output")
            finally:
                self._gen_active = False
                with self._lock:
                    self._pending -= 1

    def drain(self):
        """Wait for queued speech to finish, then close outputs.

        The timeout fires only on *stalled* progress: a long but moving
        backlog (e.g. a whole --wav file's clauses at ~1x realtime) is
        allowed to complete.
        """
        if self._pending > 0:
            log(f"draining TTS ({self._pending} clauses queued)...")
        deadline = time.monotonic() + 60
        last_progress = (self._pending, self._buffered)
        try:
            while self.is_busy() and time.monotonic() < deadline:
                progress = (self._pending, self._buffered)
                if progress != last_progress:
                    last_progress = progress
                    deadline = time.monotonic() + 60
                time.sleep(0.1)
        finally:
            if self.is_busy():
                log("TTS drain stalled; remaining speech was dropped")
            if self._wav_out:
                self._wav_out.close()
            else:
                self._stream.stop()


class LinePrinter:
    """Console line that repaints in place for partials, then finalizes.

    Final lines are also appended to each path in output_paths, and their
    translations to each path in translation_paths (with full date stamps).
    """

    def __init__(self, output_paths=(), translation_paths=()):
        self.tty = sys.stdout.isatty()
        # Partials print from the STT worker and finals from the
        # translation worker; the lock keeps their line control atomic.
        self._lock = threading.Lock()
        self.outfiles = self._open_all(output_paths, "transcripts")
        self.translation_files = self._open_all(translation_paths,
                                                "translations")

    @staticmethod
    def _open_all(paths, what):
        files = []
        for path in paths:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            files.append(open(path, "a", encoding="utf-8"))
            log(f"appending {what} to {path}")
        return files

    def partial(self, text):
        if not self.tty or not text:
            return
        with self._lock:
            width = shutil.get_terminal_size().columns - 1
            line = "… " + text
            if len(line) > width:
                line = "… " + text[-(width - 2):]
            sys.stdout.write("\r\033[K" + line)
            sys.stdout.flush()

    def final(self, text, translation=None):
        if not text:
            return
        with self._lock:
            self._final_locked(text, translation)

    def _final_locked(self, text, translation):
        now = datetime.now()
        if self.tty:
            sys.stdout.write("\r\033[K")
        print(f"[{now.strftime('%H:%M:%S')}] ", end="", flush=True)
        for word in text.split():
            print(word, end=" ", flush=True)
        print(flush=True)
        if translation:
            print(f"{'':10} → {translation}", flush=True)
        stamp = now.strftime("%Y-%m-%d %H:%M:%S")
        for outfile in self.outfiles:
            outfile.write(f"[{stamp}] {text}\n")
            outfile.flush()
        if translation:
            for outfile in self.translation_files:
                outfile.write(f"[{stamp}] {translation}\n")
                outfile.flush()


class Transcriber:
    """Whisper model (MLX or CTranslate2) plus VAD utterance segmentation."""

    def __init__(self, args):
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        self.language = args.language
        self.beam_size = args.beam_size
        self.min_silence = args.min_silence
        self.max_utterance = args.max_utterance
        self.partial_interval = args.partial_interval

        self.backend = args.backend
        if self.backend == "auto":
            if "indic-conformer" in args.model.lower():
                self.backend = "indic"
            else:
                try:
                    import mlx_whisper  # noqa: F401
                    self.backend = "mlx"
                except ImportError:
                    self.backend = "faster-whisper"

        log(f"loading model '{args.model}' with {self.backend} backend "
            f"(first run downloads it)...")
        t0 = time.time()
        if self.backend == "indic":
            if self.language not in INDIC_LANGUAGES:
                sys.exit("error: the indic backend needs --language, one of: "
                         + " ".join(INDIC_LANGUAGES))
            import torch
            from transformers import AutoModel
            self._torch = torch
            repo = args.model if "/" in args.model else INDIC_CONFORMER_REPO
            self._indic = hf_from_pretrained(AutoModel.from_pretrained,
                                             repo, trust_remote_code=True)
            self._indic_decoder = args.decoder
            self._decode(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))  # warmup
        elif self.backend == "mlx":
            import mlx_whisper
            self._mlx = mlx_whisper
            self._repo = (args.model if "/" in args.model
                          else MLX_REPOS.get(args.model))
            if self._repo is None:
                sys.exit(f"error: no known MLX conversion for '{args.model}'; "
                         f"pass a HF repo path or one of: "
                         f"{', '.join(MLX_REPOS)}")
            self._decode(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))  # warmup
        else:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(args.model, device="cpu",
                                       compute_type=args.compute_type,
                                       cpu_threads=args.cpu_threads)
            self._decode(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))  # warmup
        log(f"model ready in {time.time() - t0:.1f}s")

        self.get_speech_timestamps = get_speech_timestamps
        # Silero holds segments open for 300 ms of real silence and pads
        # ends by 100 ms; trailing_silence therefore under-reads by ~0.1 s
        # and --min-silence has an effective physical floor (DESIGN §3).
        self.vad_options = VadOptions(
            min_silence_duration_ms=300,
            speech_pad_ms=100,
        )

        self.translator = None
        translation_paths = []
        if args.translate:
            if self.backend == "indic" or self.language not in (None, "en"):
                log("warning: --translate expects English speech input")
            self.translator = Translator(args.translate_flores,
                                         args.translate_model,
                                         args.translate_beams)
            if args.output_file:
                translation_paths.append(f"log/output_stt_{args.translate}.txt")
            if args.transcript_sidecar:
                # The per-capture sidecar gets both languages.
                translation_paths.append(args.transcript_sidecar)

        self.speaker = None
        if args.speak:
            warmup = " ".join(self.translator.translate(
                "The weather is very nice today."))
            self.speaker = Speaker(args, args.tts_out, warmup_text=warmup)

        output_paths = [p for p in (args.output_file, args.transcript_sidecar)
                        if p]
        self.printer = LinePrinter(output_paths, translation_paths)
        self.buffer = np.zeros(0, dtype=np.float32)
        self.since_vad = 0.0        # seconds appended since last VAD run
        self.next_partial_at = 0.0  # monotonic time of next partial decode
        self._gated = False
        self._gate_started = 0.0

        # Pipeline threads: the segmenter (caller of feed()) never blocks
        # on inference — STT and translation each run on their own worker,
        # so endpointing of the next utterance overlaps decode/translation
        # of the previous one. Single worker per stage preserves order.
        self._pipeline_lock = threading.Lock()
        self._stt_pending = 0
        self._translate_pending = 0
        self._stt_queue = queue.Queue()
        self._translate_queue = queue.Queue()
        threading.Thread(target=self._stt_worker, daemon=True).start()
        if self.translator:
            threading.Thread(target=self._translation_worker,
                             daemon=True).start()

    def _decode(self, audio):
        if self.backend == "indic":
            wav = self._torch.from_numpy(audio).unsqueeze(0)
            with self._torch.inference_mode():
                return self._indic(wav, self.language,
                                   self._indic_decoder).strip()
        if self.backend == "mlx":
            result = self._mlx.transcribe(
                audio,
                path_or_hf_repo=self._repo,
                language=self.language,
                condition_on_previous_text=False,
            )
            return result["text"].strip()
        segments, _ = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    HALLUCINATION_MIN_WORDS = 12
    HALLUCINATION_DOMINANCE = 0.5

    def _looks_hallucinated(self, text):
        """Whisper's repetition-loop signature: one word dominating a
        long transcript (e.g. 'I I I ...' x200)."""
        words = [w.strip(".,!?").lower() for w in text.split()]
        if len(words) < self.HALLUCINATION_MIN_WORDS:
            return False
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        return max(counts.values()) / len(words) > self.HALLUCINATION_DOMINANCE

    def _post_stt(self, kind, audio):
        with self._pipeline_lock:
            self._stt_pending += 1
        self._stt_queue.put((kind, np.array(audio, copy=True)))

    def _stt_worker(self):
        while True:
            kind, audio = self._stt_queue.get()
            try:
                if kind == "partial":
                    self.printer.partial(self._decode(audio))
                    continue
                text = self._decode(audio)
                if text and self._looks_hallucinated(text):
                    log(f"dropped a likely hallucinated transcript "
                        f"({len(text.split())} words of repetition)")
                elif self.translator:
                    with self._pipeline_lock:
                        self._translate_pending += 1
                    self._translate_queue.put(text)
                else:
                    self.printer.final(text)
            except Exception as e:
                log(f"STT worker error: {e!r}")
            finally:
                with self._pipeline_lock:
                    self._stt_pending -= 1

    def _translation_worker(self):
        while True:
            text = self._translate_queue.get()
            try:
                sentences = self.translator.translate(text) if text else []
                self.printer.final(text,
                                   " ".join(sentences) if sentences else None)
                if self.speaker:
                    # Per-sentence pipelining: sentence 2 synthesizes
                    # while sentence 1 plays.
                    for sentence in sentences:
                        self.speaker.say(sentence)
            except Exception as e:
                log(f"translation worker error: {e!r}")
            finally:
                with self._pipeline_lock:
                    self._translate_pending -= 1

    def _finalize(self, audio):
        self._post_stt("final", audio)
        self.next_partial_at = time.monotonic() + self.partial_interval

    def drain_pipeline(self, timeout=60):
        """Wait for queued STT/translation work to finish (bounded)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._pipeline_lock:
                idle = (self._stt_pending == 0
                        and self._translate_pending == 0)
            if idle:
                return
            time.sleep(0.05)
        log("pipeline drain timed out; some transcripts may be missing")

    def feed(self, chunk):
        """Append float32 mono audio; emit partial and final transcripts."""
        if (self.speaker and self.speaker.gate_mic
                and self.speaker.is_audible()):
            if not self._gated:
                self._gated = True
                self._gate_started = time.monotonic()
                # A half-captured utterance can't be continued across the
                # playback gap: transcribe it if it holds real speech,
                # otherwise drop it (splicing it onto post-playback audio
                # produced fragment finals and hallucinations).
                speech = self.get_speech_timestamps(self.buffer,
                                                    self.vad_options)
                if speech and (speech[-1]["end"] - speech[0]["start"]
                               >= SAMPLE_RATE):
                    self._finalize(
                        self.buffer[speech[0]["start"]:speech[-1]["end"]])
                self.buffer = np.zeros(0, dtype=np.float32)
            return  # discard mic audio while the speakers are audible
        if self._gated:
            self._gated = False
            gated_for = time.monotonic() - self._gate_started
            if gated_for >= 0.5:
                log(f"mic was gated for {gated_for:.1f}s during speech "
                    f"playback (headphones + --tts-keep-mic avoids this)")
        self.buffer = np.concatenate([self.buffer, chunk])
        self.since_vad += len(chunk) / SAMPLE_RATE
        if self.since_vad < VAD_CHECK_SECONDS:
            return
        self.since_vad = 0.0

        speech = self.get_speech_timestamps(self.buffer, self.vad_options)
        if not speech:
            keep = int(NO_SPEECH_KEEP_SECONDS * SAMPLE_RATE)
            self.buffer = self.buffer[-keep:]
            return

        start = speech[0]["start"]
        end = speech[-1]["end"]
        trailing_silence = (len(self.buffer) - end) / SAMPLE_RATE
        speech_span = (end - start) / SAMPLE_RATE

        if trailing_silence >= self.min_silence:
            self._finalize(self.buffer[start:end])
            self.buffer = self.buffer[end:]
        elif speech_span >= self.max_utterance:
            # Speaker won't pause: flush what we have, keep a short tail
            # so we don't cut a word in half at the boundary.
            tail = int(0.2 * SAMPLE_RATE)
            self._finalize(self.buffer[start:end - tail])
            self.buffer = self.buffer[end - tail:]
        elif time.monotonic() >= self.next_partial_at:
            # Post only when the STT worker is idle: partials coalesce
            # naturally instead of queueing up behind a slow decode.
            if self.printer.tty and self._stt_pending == 0:
                self._post_stt("partial", self.buffer[start:])
            self.next_partial_at = time.monotonic() + self.partial_interval

    def flush(self):
        """Transcribe whatever speech is left in the buffer."""
        speech = self.get_speech_timestamps(self.buffer, self.vad_options)
        if speech:
            self._finalize(self.buffer[speech[0]["start"]:speech[-1]["end"]])
        self.buffer = np.zeros(0, dtype=np.float32)


def list_models():
    print("--tts-model")
    print("  mms      facebook/mms-tts-<lang>, auto-matched to the "
          "--translate language (fast, realtime; default)")
    print("  parler   ai4bharat/indic-parler-tts "
          "(higher quality, ~3 s/sentence on the GPU)")
    print("  <repo>   any Hugging Face MMS/VITS or Parler TTS repo")
    print()
    print("--tts-voice (parler engine; recommended speakers listed first)")
    for lang in sorted(PARLER_SPEAKERS):
        print(f"  {lang:4} {', '.join(PARLER_SPEAKERS[lang])}")
    print("  ...or any free-form description, e.g. "
          '"A deep male voice speaking slowly."')
    print()
    print("--translate-model")
    for name, repo in TRANSLATE_REPOS.items():
        print(f"  {name:10} {repo}")
    print()
    print("--translate languages: " + " ".join(sorted(FLORES_CODES)))


def validate_and_normalize(args, parser):
    """Resolve language codes and fail fast — before any model download.

    Normalizes args.translate to a short code (e.g. 'te') and sets
    args.translate_flores to the FLORES code IndicTrans2 needs.
    """
    if args.speak and not args.translate:
        parser.error("--speak requires --translate (it speaks the "
                     "translation)")
    if args.save_audio and args.wav:
        parser.error("--save-audio records the microphone and cannot be "
                     "combined with --wav")
    if args.tts_out and not args.speak:
        parser.error("--tts-out requires --speak")
    if args.tts_out == "auto":
        args.tts_out = datetime.now().strftime("log/tts_%Y%m%d-%H%M%S.wav")

    args.translate_flores = None
    if args.translate:
        short_by_flores = {v: k for k, v in FLORES_CODES.items()}
        if args.translate in FLORES_CODES:
            args.translate_flores = FLORES_CODES[args.translate]
        elif args.translate in short_by_flores:
            args.translate_flores = args.translate
            args.translate = short_by_flores[args.translate]
        else:
            parser.error(f"unknown --translate language '{args.translate}'; "
                         f"use one of: {' '.join(sorted(FLORES_CODES))} "
                         f"(or a FLORES code like tel_Telu)")

    if args.speak:
        # Exits with an actionable message if MMS has no voice for the
        # language — better now than after seconds of model loading.
        _, engine = resolve_tts(args.tts_model, args.translate)
        if engine == "parler" and args.tts_voice and " " not in args.tts_voice:
            known = {s for names in PARLER_SPEAKERS.values() for s in names}
            lang_speakers = PARLER_SPEAKERS.get(args.translate, [])
            if args.tts_voice not in known:
                log(f"warning: '{args.tts_voice}' is not a known "
                    f"indic-parler speaker (see --list-models)")
            elif args.tts_voice not in lang_speakers:
                log(f"warning: speaker '{args.tts_voice}' is not a "
                    f"'{args.translate}' voice; recommended: "
                    f"{', '.join(lang_speakers)}")

    if args.partial_interval <= 0:
        args.partial_interval = float("inf")
    if args.no_output_file:
        args.output_file = None
    if args.save_audio == "auto":
        args.save_audio = datetime.now().strftime(
            "log/capture_%Y%m%d-%H%M%S.wav")
    # Transcripts also go next to the saved audio, as <capture>.txt.
    args.transcript_sidecar = (os.path.splitext(args.save_audio)[0] + ".txt"
                               if args.save_audio else None)


def open_audio_writer(path):
    """Open a 16 kHz mono 16-bit WAV file for the captured mic audio."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = wave.open(path, "wb")
    writer.setnchannels(1)
    writer.setsampwidth(2)
    writer.setframerate(SAMPLE_RATE)
    log(f"saving captured audio to {path}")
    return writer


def run_mic(args, transcriber):
    import sounddevice as sd

    audio_writer = open_audio_writer(args.save_audio) if args.save_audio else None
    audio_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            log(f"audio status: {status}")
        audio_queue.put(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(BLOCK_SECONDS * SAMPLE_RATE),
        device=args.input_device,
        callback=callback,
    )
    log("listening... (Ctrl-C to stop)")
    try:
        with stream:
            while True:
                chunk = audio_queue.get()
                if audio_writer:
                    pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
                    audio_writer.writeframes(pcm.tobytes())
                transcriber.feed(chunk)
    except KeyboardInterrupt:
        log("stopping, flushing remaining audio... (Ctrl-C again to abort)")
        try:
            transcriber.flush()
            transcriber.drain_pipeline()
            if transcriber.speaker:
                transcriber.speaker.drain()
        except KeyboardInterrupt:
            log("aborted")
        log("done")
    finally:
        if audio_writer:
            audio_writer.close()


def run_wav(args, transcriber):
    """Feed an audio file through the live pipeline (replay/testing).

    Any format/rate PyAV can read is accepted and resampled to 16 kHz
    mono (WAV at any rate, stereo, mp3, m4a, ...).
    """
    from faster_whisper.audio import decode_audio
    try:
        audio = decode_audio(args.wav, sampling_rate=SAMPLE_RATE)
    except Exception as e:
        sys.exit(f"error: could not read audio from {args.wav}: {e}")

    block = int(BLOCK_SECONDS * SAMPLE_RATE)
    try:
        for i in range(0, len(audio), block):
            transcriber.feed(audio[i:i + block])
        # Simulate trailing silence so the last utterance gets emitted.
        silence = np.zeros(block, dtype=np.float32)
        for _ in range(int(transcriber.min_silence / BLOCK_SECONDS) + 2):
            transcriber.feed(silence)
    except KeyboardInterrupt:
        log("interrupted, flushing what was processed...")
    transcriber.flush()
    transcriber.drain_pipeline()
    if transcriber.speaker:
        transcriber.speaker.drain()


def main():
    p = argparse.ArgumentParser(description="Live microphone transcription "
                                            "with Whisper")
    p.add_argument("--model", default="large-v3-turbo",
                   help="Whisper model: tiny, base, small, medium, "
                        "large-v3, large-v3-turbo, distil-large-v3, a "
                        "HF repo path, or 'indic-conformer' for AI4Bharat "
                        "IndicConformer-600M-multilingual "
                        "(default: %(default)s)")
    p.add_argument("--backend", default="auto",
                   choices=["auto", "mlx", "faster-whisper", "indic"],
                   help="inference backend; auto prefers mlx (Apple GPU) "
                        "for Whisper and picks indic for indic-conformer "
                        "models (default: %(default)s)")
    p.add_argument("--decoder", default="rnnt", choices=["rnnt", "ctc"],
                   help="indic backend only: decoding head "
                        "(default: %(default)s)")
    p.add_argument("--language", default=None,
                   help="language code like 'en'; default auto-detect")
    p.add_argument("--compute-type", default="float32",
                   help="faster-whisper only: ctranslate2 compute type; "
                        "float32 is fastest on Apple Silicon "
                        "(default: %(default)s)")
    p.add_argument("--cpu-threads", type=int,
                   default=max(1, (os.cpu_count() or 4) - 2),
                   help="faster-whisper only: CPU threads for inference "
                        "(default: %(default)s)")
    p.add_argument("--beam-size", type=int, default=1,
                   help="faster-whisper only: beam size; 1 = greedy, "
                        "fastest (default: %(default)s)")
    p.add_argument("--min-silence", type=float, default=0.4,
                   help="seconds of silence that end an utterance "
                        "(default: %(default)s)")
    p.add_argument("--max-utterance", type=float, default=20.0,
                   help="force transcription after this many seconds of "
                        "continuous speech (default: %(default)s)")
    p.add_argument("--partial-interval", type=float, default=1.0,
                   help="seconds between partial-transcript updates; "
                        "0 disables partials (default: %(default)s)")
    p.add_argument("--translate", nargs="?", const="te", default=None,
                   metavar="LANG",
                   help="translate English transcripts to this Indian "
                        "language with IndicTrans2 (bare flag means 'te'; "
                        "translations also append to "
                        "log/output_stt_<LANG>.txt)")
    p.add_argument("--translate-model", default="dist-200M",
                   choices=sorted(TRANSLATE_REPOS),
                   help="IndicTrans2 variant (default: %(default)s)")
    p.add_argument("--translate-beams", type=int, default=1,
                   help="translation beam size; 1 is ~3.5x faster than 4 "
                        "with near-identical output (default: %(default)s)")
    p.add_argument("--speak", action="store_true",
                   help="speak the translated text through the speakers "
                        "(requires --translate; engine set by --tts-model)")
    p.add_argument("--tts-out", nargs="?", const="auto", default=None,
                   metavar="PATH",
                   help="write TTS audio to a WAV file instead of playing "
                        "it (for testing; default PATH is "
                        "log/tts_<timestamp>.wav)")
    p.add_argument("--tts-model", default="mms",
                   help="TTS model: 'mms' (fast, matches --translate "
                        "language), 'parler' (indic-parler-tts, higher "
                        "quality, ~3s/sentence on GPU), or a HF repo "
                        "(default: %(default)s)")
    p.add_argument("--tts-voice", default=None,
                   help="parler only: a speaker name (see --list-models) or "
                        "a full voice description; default picks the "
                        "recommended speaker for the --translate language")
    p.add_argument("--tts-keep-mic", action="store_true",
                   help="keep transcribing the mic while TTS plays "
                        "(use with headphones; otherwise TTS feeds back)")
    p.add_argument("--output-file", default="log/output_stt_en.txt",
                   help="append final transcripts to this file "
                        "(default: %(default)s)")
    p.add_argument("--no-output-file", action="store_true",
                   help="disable writing transcripts to a file")
    p.add_argument("--save-audio", nargs="?", const="auto", default=None,
                   metavar="PATH",
                   help="save captured mic audio as a 16 kHz mono WAV; "
                        "without PATH, writes log/capture_<timestamp>.wav")
    p.add_argument("--input-device", type=int, default=None,
                   help="input device index (see --list-devices)")
    p.add_argument("--list-devices", action="store_true",
                   help="list audio input devices and exit")
    p.add_argument("--list-models", action="store_true",
                   help="list known names for --tts-model, --tts-voice and "
                        "--translate-model, then exit")
    p.add_argument("--wav", default=None,
                   help="transcribe a 16 kHz mono WAV file instead of the mic")
    args = p.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if args.list_models:
        list_models()
        return

    validate_and_normalize(args, p)

    transcriber = Transcriber(args)
    if args.wav:
        run_wav(args, transcriber)
    else:
        run_mic(args, transcriber)


if __name__ == "__main__":
    main()
