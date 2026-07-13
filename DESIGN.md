# Design: live_transcribe.py

Live speech pipeline running entirely on-device (MacBook M4 Pro):
**English mic audio → transcript → Telugu translation → Telugu speech.**
Every stage uses an open-source model. This document describes the current
implementation with block diagrams; every latency figure was measured on
this machine (M4 Pro, 2026-07-12, warm models).

Authors: Radhakrishna Giduthuri, Fable 5.

## 1. Top-level pipeline

```
             ┌─────────────────────────────────────────────────────────────┐
             │                    ONE UTTERANCE'S JOURNEY                  │
             └─────────────────────────────────────────────────────────────┘

 ┌──────────┐ 100 ms  ┌──────────────┐        ┌──────────────┐  ~0.5 s   ┌─────────────┐
 │   Mic    │ blocks  │  Utterance   │ speech │  STT decode  │ (turbo,   │  Console +  │
 │ capture  ├────────►│ segmentation ├───────►│ mlx-whisper  ├──────────►│ transcript  │
 │ 16 kHz   │         │ (Silero VAD) │ audio  │  Apple GPU   │  lang=en) │   files     │
 └──────────┘         └──────────────┘        └──────┬───────┘           └─────────────┘
                    end-of-utterance detect:         │ final English text
                    0.4 s silence + ≤0.1 s poll      │
                                                     ▼
                                              ┌──────────────┐  0.14 s (dist-200M)
                                              │ Translation  │  0.35 s (1B), beam 1,
                                              │ IndicTrans2  │  per sentence, CPU
                                              └──────┬───────┘
                                                     │ Telugu sentences
                                     ┌───────────────┴───────────────┐
                                     ▼                               ▼
                              ┌─────────────┐                 ┌─────────────┐
                              │  TTS mms    │ 0.29 s for      │ TTS parler  │ ~1.0× realtime
                              │ VITS, CPU   │ 4.9 s of audio  │ fp16, MPS   │ per clause
                              │ (default)   │ (17× realtime)  │ (quality)   │ (fp32 = 0.7×)
                              └──────┬──────┘                 └──────┬──────┘
                                     └───────────────┬───────────────┘
                                                     ▼
                                              ┌─────────────┐
                                              │ Ring buffer │──► Speakers
                                              │ + jitter    │    (feedback-gates
                                              │  playback   │     the mic)
                                              └─────────────┘
```

**End-to-end latency (end of English speech → …), measured:**

| Milestone | mms TTS | parler TTS |
|---|---|---|
| Final English transcript printed | ~1.0 s | ~1.0 s |
| Telugu text printed | ~1.2–1.4 s | ~1.2–1.4 s |
| Telugu voice starts | **~1.7 s** | **~4–5 s** (first clause must fully generate) |
| Playback smoothness | gap-free | gap-free (clause-chained) |

While speaking, a partial English transcript repaints in place every ~1 s
(one extra STT decode per interval, same ~0.5 s cost, absorbed between mic
blocks).

## 2. Threading model

Six independent execution contexts — one per functional block — connected
by ordered queues, so no model inference ever blocks audio I/O or
segmentation.

```
 PortAudio input        Main thread (segmenter)      STT worker            Translation worker
 ┌────────────────┐   ┌──────────────────────┐   ┌────────────────┐   ┌──────────────────────┐
 │ InputStream    │   │ chunk = queue.get()  │   │ decode partial │   │ translate sentences  │
 │ callback 100ms ├──►│ feed(): mic gate,    ├──►│  └► console    ├──►│  └► print final +    │
 │ └► audio_queue │   │ VAD, endpointing —   │   │ decode final   │   │     say() clauses ──┐│
 └────────────────┘   │ never blocks         │   │ + halluc guard │   └─────────────────────┼┘
                      └──────────────────────┘   └────────────────┘                         │
                                                  TTS worker                PortAudio output▼
                                                 ┌────────────────────┐   ┌─────────────────────┐
                                                 │ synthesize clause  ├──►│ ring buffer +       │
                                                 │ (fp16 parler/mms)  │   │ jitter playback     │
                                                 └────────────────────┘   └─────────────────────┘
```

- **The segmenter never blocks on inference:** endpointing of utterance
  N+1 proceeds while N is being decoded, translated, and spoken. A single
  worker per stage preserves output order; per-stage pending counters
  (lock-protected) drive `drain_pipeline()` at shutdown.
- **Partials coalesce naturally:** the segmenter posts a partial decode
  only when the STT worker is idle (and stdout is a tty), so a slow decode
  can never queue a backlog of stale partials.
- **Mic gating is audibility-based:** the gate engages only while the
  speakers are actually emitting sound (`Speaker.is_audible()`), not
  during parler's silent generation lead — and each gate window logs how
  long the mic was off. At the gate transition the rolling buffer is
  finalized (if it holds ≥1 s of speech) or cleared, never spliced across
  the playback gap. In gated (speakerphone) mode, speech during playback
  is still intentionally discarded — headphones + `--tts-keep-mic` is the
  zero-loss configuration for continuous dictation.
- **Hallucination guard:** finals where one word dominates a long
  transcript (Whisper's repetition-loop signature) are dropped with a log
  line instead of being translated and spoken.

## 3. Utterance segmentation (feed loop)

The rolling buffer plus Silero VAD (bundled ONNX in faster-whisper — one
pass over a 20 s buffer costs only **~25 ms**, so polling every 0.1 s is
nearly free):

```
  mic chunk (100 ms)
        │
        ▼
  TTS playing? ──yes──► discard (feedback gate; --tts-keep-mic disables)
        │no
        ▼
  append to rolling buffer ──► VAD every 0.1 s (~20–25 ms/pass)
        │
        ├─ no speech anywhere ────────► keep last 1 s, wait
        │
        ├─ trailing silence ≥ 0.4 s ──► FINALIZE utterance   ─┐
        │                                                     │ cut buffer
        ├─ speech span ≥ 20 s ────────► FINALIZE (forced)    ─┘ at speech end
        │
        └─ else, every ≥1 s ──────────► partial decode, repaint console line
```

Only detected speech ever reaches the STT model — Whisper never sees
silence, which avoids its silence-hallucination failure mode and keeps
decode cost bounded.

**Endpointing contributes a fixed ~0.45 s** (0.4 s required silence +
≤0.1 s poll cadence) to every utterance's latency. This is a deliberate
trade against splitting sentences at brief pauses.

## 4. STT block

Three interchangeable backends behind one `_decode(audio) -> text`:

```
             ┌────────────────────────────────────────────────┐
   audio ───►│ mlx  (default)   Whisper large-v3-turbo, GPU   │ ~0.5 s  (lang pinned)
             │                                                │ ~0.95 s (lang auto-detect)
             │ faster-whisper   Whisper on CPU (CTranslate2)  │ ~2.6 s  (float32 > int8!)
             │ indic            IndicConformer-600M, 22 langs │ ~0.15 s (rnnt decoder)
             └────────────────────────────────────────────────┘
```

Two non-obvious properties, both measured:
- **Decode cost is flat in utterance length** (0.53 s for 2 s of audio,
  0.51 s for 10 s): Whisper pads every input to a 30-second mel window, so
  short utterances pay the same encoder cost as long ones.
- Pinning `--language en` roughly **halves** mlx decode latency vs
  auto-detection.

## 5. Translation block

```
  English text ─► sentence split (regex) ─► IndicProcessor preprocess
              ─► IndicTrans2 generate (beam 1) ─► postprocess ─► [te sentences]
```

Measured per sentence, CPU: **dist-200M 0.14 s, 1B 0.35 s** at beam 1.
Beam 4 is ~3.5× slower and produced identical output in tests — hence
`--translate-beams` defaults to 1. Sentences are batched through the model
in one `generate` call, then handed to TTS individually so synthesis of
sentence 2 overlaps playback of sentence 1.

## 6. TTS block and the jitter-buffered ring buffer

The core constraint: **Parler generates at ~1.0× realtime in fp16 on MPS
(0.7× in fp32), while MMS generates at ~17× realtime on CPU.** Anything
≤1× cannot stream chunk-by-chunk without eventual underruns — a jitter
buffer of B seconds drains after B/(1−g) seconds at generation rate g.
(The stock `ParlerTTSStreamer` also re-decodes the whole codec sequence
per chunk, costing another 30–45%; it was removed.)

Design: split text into clauses and generate each **completely** before it
enters the ring buffer. A pause can then only ever fall on a clause
boundary — where it sounds like natural phrasing — never mid-word.

```
  "ఈ రోజు వాతావరణం చాలా బాగుంది, సూర్యరశ్మిని చూసి నేను సంతోషిస్తున్నాను."
        │  clause split at , ; : ।  (12–48 chars, word-boundary fallback)
        ▼
  [clause 1] [clause 2] [clause 3]        TTS worker (sequential)
       │          │          │
       ▼          ▼          ▼            each clause: full synthesis,
  ┌─────────────────────────────────┐     ~1.0× realtime (parler fp16)
  │ ring buffer (float32 samples)   │     or ~17× (mms)
  └───────────────┬─────────────────┘
                  ▼
  playback callback:  not playing ─► wait for 1 s buffered (while generating)
                      playing     ─► drain buffer; empty ─► pause & rebuffer
```

Timeline for a two-clause sentence with parler (measured behavior,
0 mid-speech pauses):

```
  wall time ──►  0        d₁        2·d₁      d₁+d₂     ...
  worker:        ├─gen clause 1─┤├─gen clause 2─┤
  playback:                     ├─play clause 1─┤├─play clause 2─┤
                                ▲
                                └─ voice onset ≈ duration of clause 1 (~2–4 s)
```

Because generation ≈ playback speed, clause N+1 finishes generating just
as clause N finishes playing. The trade is explicit: **voice onset can
never beat the first clause's duration** at 1.0× generation. Making the
first clause shorter would start sooner but open a gap right after it
(short clause plays out faster than the longer next clause generates).
MMS has no such constraint — onset ~0.3 s.

**Feedback gating:** `Speaker.is_busy()` (queue pending ∨ generating ∨
buffer non-empty, +0.3 s tail) makes `feed()` discard mic input, so the
app never transcribes its own voice. Verified live: spoken English through
the room's speakers was translated and spoken back with zero feedback
lines.

## 7. Outputs

```
  final English ──► console  [HH:MM:SS] text          (word-by-word, flushed)
              ├──► log/output_stt_en.txt              (default, --output-file)
              └──► <capture>.txt sidecar              (with --save-audio)
  translation ───► console  "         → తెలుగు"
              ├──► log/output_stt_<lang>.txt
              └──► <capture>.txt sidecar (both languages, interleaved)
  mic audio ─────► <capture>.wav                      (with --save-audio)
  TTS audio ─────► speakers (also in --wav replay), or a WAV via --tts-out
```

Partials are console-only and suppressed when stdout is not a tty.

## 8. Model loading (startup, one-time)

| Block | Load + warmup |
|---|---|
| mlx-whisper large-v3-turbo | ~1 s (cached) |
| IndicTrans2 dist-200M / 1B | ~1.5–4 s |
| MMS-TTS | ~1 s |
| indic-parler-tts (fp16, MPS) | **12–19 s** (Metal warmup alongside mlx; warmup uses a representative-length clause — short warmups leave the first real generation ~30% slow) |

## 9. Testing strategy

- `--wav` mode drives the identical `feed()` path as the mic; TTS writes
  to a file instead of playing. macOS `say -v Geeta` synthesizes Telugu
  test speech; `[[slnc 1200]]` inserts pauses to exercise segmentation.
- **Round-trip validation:** TTS output → IndicConformer STT → compare
  with the input text (used to validate both TTS engines and fp16).
- Live loop: `say` plays English through the speakers while the app
  listens — validates the whole pipeline plus the feedback gate.
