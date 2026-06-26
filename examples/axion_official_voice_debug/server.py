#!/usr/bin/env python3
"""Pipecat official-like voice debug using OpenAI-compatible STT, LLM, and TTS."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PIPECAT_ROOT = Path(__file__).resolve().parents[2]

if str(PIPECAT_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPECAT_ROOT))

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(PROJECT_ROOT / ".env", override=False)
load_dotenv(override=False)

DEFAULT_BASE_URL = "https://api.zhizengzeng.com/v1"


def env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def require_env(*names: str) -> str:
    value = env_first(*names)
    if value:
        return value
    joined = ", ".join(names)
    raise RuntimeError(f"missing required env: {joined}")


def is_zhizengzeng_base_url(value: str) -> bool:
    return "api.zhizengzeng.com" in value.strip().lower()


transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


def resolve_language(value: str) -> Language:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "zh": Language.ZH,
        "zh_cn": Language.ZH,
        "cn": Language.ZH,
        "en": Language.EN,
        "en_us": Language.EN,
    }
    return aliases.get(normalized, Language.ZH)


def build_services():
    base_url = env_first("AXION_PIPECAT_LLM_BASE_URL", default=DEFAULT_BASE_URL)
    api_key_names = ["AXION_PIPECAT_LLM_API_KEY"]
    if is_zhizengzeng_base_url(base_url):
        api_key_names.extend(["ZHIZENGZENG_API_KEY", "ZHIZENZENG_API_KEY"])
    api_key_names.extend(["OPENAI_API_KEY", "DASHSCOPE_SECRET_KEY", "DASHSCOPE_API_KEY"])
    api_key = require_env(*api_key_names)
    stt_model = env_first("AXION_PIPECAT_OFFICIAL_STT_MODEL", default="whisper-1")
    llm_model = env_first("AXION_PIPECAT_LLM_MODEL", "OPENAI_MODEL", default="chat-latest")
    tts_model = env_first("AXION_PIPECAT_OFFICIAL_TTS_MODEL", default="tts-1")
    tts_voice = env_first("AXION_PIPECAT_OFFICIAL_TTS_VOICE", default="alloy")
    stt_language = resolve_language(env_first("AXION_PIPECAT_OFFICIAL_STT_LANGUAGE", default="zh"))
    stt_prompt = env_first(
        "AXION_PIPECAT_OFFICIAL_STT_PROMPT",
        default="This audio may contain both Chinese and English.",
    )
    system_prompt = env_first(
        "AXION_PIPECAT_SYSTEM_PROMPT",
        default=(
            "You are a helpful assistant in a voice conversation. "
            "Your responses will be spoken aloud, so avoid markdown, emoji, bullets, "
            "and other formatting that does not sound natural when spoken. "
            "Keep replies concise, friendly, and easy to listen to."
        ),
    )

    logger.info(
        "official voice config: base_url={} stt_model={} stt_language={} llm_model={} tts_model={} tts_voice={}",
        base_url,
        stt_model,
        stt_language,
        llm_model,
        tts_model,
        tts_voice,
    )

    stt = OpenAISTTService(
        api_key=api_key,
        base_url=base_url,
        settings=OpenAISTTService.Settings(
            model=stt_model,
            language=stt_language,
            prompt=stt_prompt,
        ),
    )
    llm = OpenAILLMService(
        api_key=api_key,
        base_url=base_url,
        settings=OpenAILLMService.Settings(
            model=llm_model,
            system_instruction=system_prompt,
        ),
        retry_timeout_secs=30.0,
        retry_on_timeout=True,
    )
    tts = OpenAITTSService(
        api_key=api_key,
        base_url=base_url,
        settings=OpenAITTSService.Settings(
            model=tts_model,
            voice=tts_voice,
        ),
    )
    return stt, llm, tts


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("official voice debug: starting bot")
    stt, llm, tts = build_services()

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("official voice debug: client connected")
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("official voice debug: client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
