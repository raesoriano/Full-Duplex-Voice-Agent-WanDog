"""Fweddie -- resample the mic stream between the capture rate and 16 kHz.

Why this exists
---------------
The USB capture card on this Pi (``hw:2,0``, PortAudio index 0) accepts
exactly ONE capture rate: 48000 Hz. Every other rate we tried
(8000/16000/22050/32000/44100/96000) fails with ``Errno -9997 Invalid sample
rate``. That is a property of the hardware, not of PortAudio or ALSA.

Both consumers downstream want 16 kHz:

  * faster-whisper (the ASR) is trained on 16 kHz audio;
  * Silero VAD only accepts 8000 or 16000 Hz (see ``silero.py``).

So we cannot capture at 16 kHz, and we cannot feed 48 kHz to VAD or ASR.
The fix is to capture at 48000 and convert to 16000 in the pipeline, which is
what this processor does. It sits directly after the input transport and
before everything that cares about the rate.

``soxr`` keeps internal history between chunks, so there are no clicks at
chunk boundaries.
"""

from __future__ import annotations

from loguru import logger

from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AudioResampleProcessor(FrameProcessor):
    """Convert ``InputAudioRawFrame`` audio from ``in_rate`` to ``out_rate``.

    Every other frame passes through untouched. The frame's own
    ``sample_rate`` is rewritten so downstream processors (VAD, ASR) see the
    rate the audio is actually in -- this matters because Pipecat reads the
    rate off the frame, not off a global config.
    """

    def __init__(self, in_rate: int, out_rate: int, **kwargs):
        super().__init__(**kwargs)
        self._in_rate = in_rate
        self._out_rate = out_rate
        # quality="QQ" is the cheapest soxr preset. The Pi is doing VAD, ASR
        # and TTS at once; resampling quality above this is inaudible for
        # speech going into Whisper.
        self._resampler = SOXRStreamAudioResampler(quality="QQ")
        self._passthrough = in_rate == out_rate
        if self._passthrough:
            logger.info(
                "Resampler: in == out ({} Hz), running as a no-op", in_rate
            )
        else:
            logger.info("Resampler: {} Hz -> {} Hz", in_rate, out_rate)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame) and not self._passthrough:
            audio = await self._resampler.resample(
                frame.audio, self._in_rate, self._out_rate
            )
            out = InputAudioRawFrame(
                audio=audio,
                sample_rate=self._out_rate,
                num_channels=frame.num_channels,
            )
            out.pts = frame.pts
            await self.push_frame(out, direction)
            return

        await self.push_frame(frame, direction)
