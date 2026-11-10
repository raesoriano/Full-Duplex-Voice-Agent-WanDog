"""Live end-to-end check: Fweddie's exact LLM wiring against the DGX cluster.

Builds pipecat's OpenAILLMService the same way agent.py does (key from the OS
keychain via config.resolve_llm_api_key, thinking disabled for Qwen3.8), feeds
one user turn, and asserts the model returns non-empty text.

Run:  .venv/bin/python test_llm_live.py
"""

import asyncio
import sys

import config
from pipecat.frames.frames import LLMContextFrame, StartFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService


class Sink(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.texts: list[str] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        text = getattr(frame, "text", None)
        if text:
            self.texts.append(text)


def build_llm() -> OpenAILLMService:
    """Mirror of the LLM construction in agent.py."""
    llm_settings = dict(
        model=config.LLM_MODEL,
        system_instruction="You are Fweddie, a voice assistant. Answer in one short sentence.",
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )
    if not config.LLM_THINKING:
        llm_settings["extra"] = {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
        }
    return OpenAILLMService(
        api_key=config.resolve_llm_api_key(),
        base_url=config.LLM_BASE_URL,
        settings=OpenAILLMService.Settings(**llm_settings),
    )


async def main() -> int:
    llm = build_llm()
    sink = Sink()
    llm._output_processor = sink
    llm._next = sink
    sink._prev = llm
    await llm.start(StartFrame())
    try:
        ctx = LLMContext()
        ctx.add_message({"role": "user", "content": "What is 2+2? Reply in one word."})
        await llm.process_frame(LLMContextFrame(ctx), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(8)
        # Read the sink's input queue (its process task is not started in this
        # harness; a real pipeline runs it).
        q = sink._FrameProcessor__input_queue
        for item in list(q._queue):
            frame = item[2][0] if isinstance(item, tuple) and item[2] else item
            text = getattr(frame, "text", None)
            if text:
                sink.texts.append(text)
    finally:
        await llm.cleanup()

    reply = "".join(sink.texts).strip()
    print(f"model:      {config.LLM_MODEL}")
    print(f"endpoint:   {config.LLM_BASE_URL}")
    print(f"key source: OS keychain (service 'onit')")
    print(f"reply:      {reply!r}")
    if not reply:
        print("FAIL: empty reply")
        return 1
    print("PASS: LLM returned text through pipecat's OpenAILLMService")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
