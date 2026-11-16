"""Fweddie -- streaming ASR: partial text while the user is still dictating.

Why this file exists
--------------------
``WhisperSTTService`` is a *segmented* service: it buffers the whole utterance
and only starts transcribing after the VAD reports the user stopped speaking.
On this Pi that means

    speech ends  ->  (VAD 0.7 s)  ->  (whisper 4.9 s)  ->  transcript

i.e. the words arrive ~5.5 s after the user stopped talking, long after the
turn has closed. That is the root cause of "it answers the previous question".

``sherpa-onnx`` ships a genuinely streaming transducer (zipformer) that decodes
20 ms chunks as they arrive. Measured here:

    zipformer streaming, RTF 0.10 -> first partial word ~1.0 s after speech
    starts, every later word ~0.32 s after it is spoken, and the final text is
    ready the instant the audio ends (no post-processing wait).

So this service emits:

  * ``InterimTranscriptionFrame`` continuously while the user talks -- the
    dashboard can show the words appear as they are spoken, and turn-taking
    strategies can react before the utterance is over;
  * one ``TranscriptionFrame(finalized=True)`` at the end of the utterance.

Why a second, non-streaming pass
--------------------------------
The streaming zipformer is fast but its accuracy on this mic is poor on
numbers: "what is 2 plus 2?" decodes as "WHAT IS TOO PLUS". faster-whisper
``tiny.en`` gets it exactly right in 0.86 s. So by default the streaming model
produces the live partials and faster-whisper produces the authoritative final
text -- both from the same audio buffer. That is the ``final_pass`` argument.
Set it to ``None`` (``FWEDDIE_ASR_FINAL=none``) for a pure streaming path.

Cost of the hybrid, measured: speech end -> final transcript ~1.6 s instead of
~5.5 s, and the user sees their words appear from ~1.0 s into the utterance.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601

try:
    import sherpa_onnx
except ModuleNotFoundError as e:  # pragma: no cover - install-time problem
    raise ImportError(
        "streaming ASR needs sherpa-onnx: .venv/bin/pip install sherpa-onnx"
    ) from e

# The zipformer is trained on 16 kHz mono. The pipeline resamples 48 kHz mic
# audio down to this rate before the STT stage (see config.ASR_SAMPLE_RATE).
SAMPLE_RATE = 16000


class SherpaStreamingSTTService(STTService):
    """Streaming speech-to-text with optional accurate final pass.

    Parameters:
        model_dir: Directory holding the sherpa-onnx streaming zipformer files
            (``tokens.txt``, ``encoder-*.onnx``, ``decoder-*.onnx``,
            ``joiner-*.onnx``). int8 encoder/joiner are used when present.
        num_threads: ONNX Runtime threads for the encoder. The Pi has 4 cores
            and must also run VAD, the LLM client and Piper, so 2 is the
            default.
        final_pass: ``"whisper"`` to re-transcribe each finished segment with
            faster-whisper (accurate final text), or ``None`` to use the
            streaming model's own output as the final text.
        final_model: faster-whisper model name for the final pass.
        pre_roll_secs: Audio kept from *before* the VAD fired. VAD only reports
            speech after ~0.25 s of it, so without this the first syllable is
            cut off -- the streaming decoder would start mid-word.
        interim_interval: Minimum seconds between two ``InterimTranscriptionFrame``
            pushes. The decoder produces a new hypothesis every ~0.32 s; this
            throttles bursts without hiding the words.
    """

    def __init__(
        self,
        *,
        model_dir: str | Path,
        num_threads: int = 2,
        final_pass: str | None = "whisper",
        final_model: str = "tiny.en",
        final_compute: str = "int8",
        language: str = "en",
        pre_roll_secs: float = 1.0,
        interim_interval: float = 0.15,
        ttfs_p99_latency: float | None = None,
        settings: STTSettings | None = None,
        **kwargs,
    ):
        super().__init__(
            # model/language must be set (not NOT_GIVEN) or Pipecat logs a
            # settings-validation error at startup.
            settings=settings
            or STTSettings(model=Path(model_dir).name, language=language),
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )
        self._model_dir = Path(model_dir)
        self._num_threads = num_threads
        self._final_pass = (final_pass or "").lower() or None
        self._final_model_name = final_model
        self._final_compute = final_compute
        self._language = language
        self._pre_roll_bytes = int(pre_roll_secs * SAMPLE_RATE) * 2
        self._interim_interval = interim_interval

        self._recognizer = None
        self._whisper = None
        self._segment_task: asyncio.Task | None = None
        self._segment_queue: asyncio.Queue = asyncio.Queue()

        # Per-utterance state.
        self._stream = None
        self._speaking = False
        self._preroll = bytearray()
        self._segment_audio = bytearray()
        self._last_partial = ""
        self._last_partial_at = 0.0
        self._segment_count = 0
        self._transcribing_segment = 0

        self._interim_count = 0
        self._final_count = 0

    # ------------------------------------------------------------------ setup

    def _resolve_model_files(self) -> dict:
        """Pick the int8 weights when the model directory has them.

        int8 is ~3.7x smaller and measurably faster here (0.27 s vs 0.40 s for
        a 1.8 s utterance) with identical output, so it is not a compromise.
        """
        d = self._model_dir
        if not d.is_dir():
            raise FileNotFoundError(
                f"streaming ASR model not found: {d}\n"
                "Download it with:\n"
                "  mkdir -p models && cd models && curl -LO "
                "https://github.com/k2-fsa/sherpa-onnx/"
                "releases/download/asr-models/"
                "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2\n"
                "  tar xf sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
            )

        def pick(pattern: str) -> str:
            matches = sorted(d.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"{d} has no file matching {pattern}")
            # Prefer the int8 variant when both exist.
            for m in matches:
                if ".int8." in m.name:
                    return str(m)
            return str(matches[0])

        files = {
            "tokens": str(d / "tokens.txt"),
            "encoder": pick("encoder-*.onnx"),
            "decoder": pick("decoder-*.onnx"),
            "joiner": pick("joiner-*.onnx"),
        }
        for name, path in files.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"streaming ASR: missing {name} at {path}")
        return files

    async def start(self, frame: StartFrame):
        """Load the models off the event loop and start the segment worker."""
        await super().start(frame)

        files = self._resolve_model_files()
        logger.info(
            "Streaming ASR: {} (encoder {}, {} threads)",
            self._model_dir.name,
            Path(files["encoder"]).name,
            self._num_threads,
        )

        def _load_recognizer():
            return sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=files["tokens"],
                encoder=files["encoder"],
                decoder=files["decoder"],
                joiner=files["joiner"],
                num_threads=self._num_threads,
                sample_rate=SAMPLE_RATE,
                feature_dim=80,
                decoding_method="greedy_search",
                # Segmentation is the VAD's job here, not the recognizer's:
                # endpointing inside a turn would reset the stream mid-sentence.
                enable_endpoint_detection=False,
            )

        self._recognizer = await asyncio.to_thread(_load_recognizer)

        if self._final_pass == "whisper":
            self._whisper = await asyncio.to_thread(self._load_whisper)
            if self._whisper is None:
                logger.warning("Streaming ASR: final pass unavailable, using streaming text")

        self._segment_task = self.create_task(self._segment_worker())
        logger.info(
            "Streaming ASR ready (final pass: {})",
            self._final_model_name if self._whisper else "streaming text only",
        )

    def _load_whisper(self):
        """Load faster-whisper for the final pass. Runs in a worker thread."""
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(
                self._final_model_name, device="cpu", compute_type=self._final_compute
            )
            # Warm the graph so the first real turn is not the slow one.
            model.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32), language=self._language)
            return model
        except Exception as e:  # noqa: BLE001 -- degrade, do not crash the agent
            logger.warning("Streaming ASR: could not load {}: {}", self._final_model_name, e)
            return None

    async def cleanup(self):
        if self._segment_task is not None:
            await self.cancel_task(self._segment_task)
            self._segment_task = None
        await super().cleanup()

    # ------------------------------------------------------------ STT contract

    async def run_stt(self, audio: bytes):  # pragma: no cover - never used
        """Unused: this service overrides ``process_audio_frame`` entirely.

        ``STTService`` declares ``run_stt`` abstract and the segmented base
        calls it per utterance. Streaming has no per-utterance request, so the
        audio path is taken over below and this stays empty.
        """
        return
        yield  # makes this an async generator

    @property
    def supports_ttfs(self) -> bool:
        """Streaming text arrives continuously, so speech-end -> final is short.

        ``SpeechTimeoutUserTurnStopStrategy`` reads this through
        ``STTMetadataFrame`` as a safety net for the case where the final
        transcript never arrives.
        """
        return True

    # ------------------------------------------------------------- audio path

    async def process_audio_frame(self, frame: InputAudioRawFrame, direction):
        """Feed the streaming decoder chunk by chunk, and keep a pre-roll buffer."""
        if self._muted or not self.is_usable or self._recognizer is None:
            return

        self._user_id = getattr(frame, "user_id", "") or ""
        chunk = frame.audio
        if not chunk:
            return

        # Always keep the last pre_roll_secs of audio: the VAD fires ~0.25 s
        # after speech begins and we do not want to lose that first syllable.
        self._preroll += chunk
        if len(self._preroll) > self._pre_roll_bytes:
            del self._preroll[: len(self._preroll) - self._pre_roll_bytes]

        if not self._speaking:
            return

        self._segment_audio += chunk
        await self._decode(chunk, emit_interim=True)

    async def _decode(self, pcm: bytes, *, emit_interim: bool) -> str:
        """Push PCM into the recognizer, decode, and optionally emit a partial."""
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self._stream.accept_waveform(SAMPLE_RATE, samples)
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        text = (self._recognizer.get_result(self._stream) or "").strip()

        if emit_interim and text and text != self._last_partial:
            now = time.monotonic()
            if now - self._last_partial_at >= self._interim_interval:
                self._last_partial = text
                self._last_partial_at = now
                self._interim_count += 1
                await self.push_frame(
                    InterimTranscriptionFrame(
                        text=_tidy(text),
                        user_id=self._user_id,
                        timestamp=time_now_iso8601(),
                        language=self._language,
                    )
                )
        return text

    # ------------------------------------------------------------- turn edges

    async def process_frame(self, frame: Frame, direction):
        """Drive the streaming segment from the VAD's own start/stop frames.

        ``super().process_frame`` handles the audio frames, muting, settings and
        the TTFB bookkeeping -- and it pushes the VAD frames downstream, so they
        are only *observed* here, never re-pushed.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._begin_segment()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._end_segment()

    def _begin_segment(self) -> None:
        """Start a fresh decoder stream, primed with the pre-roll audio."""
        self._segment_count += 1
        self._stream = self._recognizer.create_stream()
        self._speaking = True
        self._last_partial = ""
        self._last_partial_at = 0.0
        self._segment_audio = bytearray(self._preroll)

        if self._segment_audio:
            # Decode the pre-roll synchronously so the first partial already
            # includes the syllable the VAD needed to hear before firing.
            samples = (
                np.frombuffer(bytes(self._segment_audio), dtype=np.int16).astype(np.float32)
                / 32768.0
            )
            self._stream.accept_waveform(SAMPLE_RATE, samples)
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)

    def _end_segment(self) -> None:
        """Flush the decoder and queue the segment for the final transcript."""
        if not self._speaking or self._stream is None:
            return
        self._speaking = False

        # Nothing left to decode? Then there was no speech in this segment.
        stream = self._stream
        self._stream = None
        stream.input_finished()
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)
        streaming_text = (self._recognizer.get_result(stream) or "").strip()

        audio = bytes(self._segment_audio)
        self._segment_audio = bytearray()
        self._record_stt_audio_usage(audio)
        self._segment_queue.put_nowait((audio, streaming_text, self._segment_count))

    # -------------------------------------------------------- final transcript

    async def _segment_worker(self):
        """Transcribe finished segments one at a time, in order."""
        while True:
            item = await self._segment_queue.get()
            if item is None:
                return
            audio, streaming_text, seg_id = item
            self._transcribing_segment = seg_id
            try:
                text = await self._final_text(audio, streaming_text)
            except Exception as e:  # noqa: BLE001
                await self.push_error(f"{self}: final transcript failed: {e}", exception=e)
                text = streaming_text

            text = (text or "").strip()
            if not text:
                # No words: emit nothing. The user aggregator then treats this
                # as a turn without a transcript rather than injecting an empty
                # user message (which is what produced "please repeat that").
                logger.debug("Streaming ASR: segment {} had no speech", seg_id)
                continue

            self._final_count += 1
            await self.push_frame(
                TranscriptionFrame(
                    text=_tidy(text),
                    user_id=self._user_id,
                    timestamp=time_now_iso8601(),
                    language=self._language,
                    finalized=True,
                )
            )

    async def _final_text(self, audio: bytes, streaming_text: str) -> str:
        """The authoritative text for one utterance.

        The streaming hypothesis is used when it already agrees with nothing
        better, and as the fallback whenever the final pass yields no words.
        """
        if self._whisper is None or not audio:
            return streaming_text

        # No words from the streaming model means this segment held no speech
        # (a cough, a keyboard click, the speaker's own tail). Do NOT hand that
        # to whisper: faster-whisper is trained to always produce something and
        # invents whole sentences out of noise. Measured here, a noise-only
        # segment decoded as "Well did I not? What's in the nog?" -- which then
        # went into the prompt as if the user had said it.
        if not streaming_text:
            logger.debug("Streaming ASR: no words in segment, skipping the final pass")
            return ""

        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        # The last word is routinely clipped when audio stops abruptly.
        samples = np.concatenate([samples, np.zeros(int(SAMPLE_RATE * 0.2), dtype=np.float32)])

        def _run() -> str:
            segments, _ = self._whisper.transcribe(
                samples,
                language=self._language,
                beam_size=1,          # greedy: 0.9 s instead of 1.6 s on the Pi
                condition_on_previous_text=False,
                vad_filter=False,     # the VAD already cut this segment
            )
            return " ".join(s.text for s in segments).strip()

        text = await asyncio.to_thread(_run)
        logger.debug("Streaming ASR: streaming={!r} final={!r}", streaming_text, text)
        return text or streaming_text


def _tidy(text: str) -> str:
    """Lowercase the zipformer's ALL-CAPS output and capitalise the first letter.

    The streaming model emits "WHAT IS TOO PLUS" with no punctuation. The
    lowercase form reads as normal text in the dashboard and in the LLM prompt.
    """
    t = " ".join(text.split())
    if not t:
        return t
    # Only touch text that is entirely upper case -- a whisper final already
    # has proper casing and must not be flattened.
    if t.upper() == t:
        t = t.lower()
        t = t[0].upper() + t[1:]
    return t
