#!/usr/bin/env python3
"""Pipecat voice debug demo using Qwen ASR/TTS and Glenclaw LLM."""

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

from axion_voice import QwenASRProcessor, QwenTTSProcessor
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMRunFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnMessageAddedMessage,
    UserTurnStoppedMessage,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

load_dotenv(PROJECT_ROOT / ".env", override=False)
load_dotenv(override=False)

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


class GlenclawOpenAILLMService(OpenAILLMService):
    """OpenAI-compatible LLM service tuned for Glenclaw's chat endpoint."""

    supports_developer_role = False


class ClientCompatibleProtobufSerializer(ProtobufFrameSerializer):
    """Keep interruption handling server-side without exposing raw frames to the web client."""

    async def serialize(self, frame: Frame):
        if isinstance(frame, InterruptionFrame):
            return None
        return await super().serialize(frame)


class ClientFrameCompatibilityFilter(FrameProcessor):
    """Drop internal frames the bundled browser client cannot deserialize."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(
            frame,
            (
                VADUserStartedSpeakingFrame,
                VADUserStoppedSpeakingFrame,
            ),
        ):
            return

        await self.push_frame(frame, direction)


def build_local_vad_analyzer() -> SileroVADAnalyzer:
    return SileroVADAnalyzer(
        params=VADParams(
            confidence=0.72,
            start_secs=0.15,
            stop_secs=0.25,
            min_volume=0.70,
        )
    )


def build_transport_params() -> dict[str, callable]:
    return {
        "eval": lambda: EvalTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        "twilio": lambda: FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        "websocket": lambda: FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=ClientCompatibleProtobufSerializer(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }


def build_services() -> tuple[QwenASRProcessor, OpenAILLMService, QwenTTSProcessor]:
    glenclaw_base_url = env_first("GLENCLAW_BASE_URL", default="https://api.glenclaw.com")
    llm_base_url = env_first(
        "AXION_PIPECAT_LLM_BASE_URL",
        default=f"{glenclaw_base_url.rstrip('/')}/api/v1/openai",
    )
    llm_model = env_first("AXION_PIPECAT_LLM_MODEL", "OPENAI_MODEL", default="chat")
    llm_api_key = require_env("GLENCLAW_CLOUD_SK")

    speech_api_key = require_env(
        "AXION_PIPECAT_SPEECH_API_KEY",
        "GLENCLAW_SPEECH_API_KEY",
        "DASHSCOPE_SECRET_KEY",
        "DASHSCOPE_API_KEY",
    )
    asr_endpoint = env_first(
        "AXION_PIPECAT_ASR_WS_URL",
        "GLENCLAW_ASR_WS_URL",
        default=QwenASRProcessor.ENDPOINT,
    )
    tts_endpoint = env_first(
        "AXION_PIPECAT_TTS_WS_URL",
        "GLENCLAW_TTS_WS_URL",
        default=QwenTTSProcessor.ENDPOINT,
    )
    asr_model = env_first("AXION_PIPECAT_ASR_MODEL", default="qwen3-asr-flash-realtime")
    tts_model = env_first("AXION_PIPECAT_TTS_MODEL", default="qwen3-tts-flash-realtime")
    tts_voice = env_first("AXION_PIPECAT_TTS_VOICE", default="Chelsie")
    tts_language = "Chinese"
    system_prompt = env_first(
        "AXION_PIPECAT_SYSTEM_PROMPT",
        default=(
            "You are a helpful assistant in a voice conversation. "
            "Keep your spoken replies concise, natural, and easy to listen to. "
            "Avoid markdown, bullet points, code fences, and URLs unless the user asks."
        ),
    )

    logger.info(
        "voice debug config: llm_provider=glenclaw-openai llm_base_url={} llm_model={} asr_model={} tts_model={} tts_voice={}",
        llm_base_url,
        llm_model,
        asr_model,
        tts_model,
        tts_voice,
    )

    stt = QwenASRProcessor(
        api_key=speech_api_key,
        endpoint=asr_endpoint,
        model=asr_model,
        sample_rate=16000,
        server_vad=True,
        silence_duration_ms=400,
        preroll_secs=0.4,
    )
    llm = GlenclawOpenAILLMService(
        api_key=llm_api_key,
        base_url=llm_base_url,
        settings=GlenclawOpenAILLMService.Settings(
            model=llm_model,
            system_instruction=system_prompt,
        ),
        retry_timeout_secs=30.0,
        retry_on_timeout=True,
    )
    tts = QwenTTSProcessor(
        api_key=speech_api_key,
        endpoint=tts_endpoint,
        model=tts_model,
        voice=tts_voice,
        sample_rate=24000,
        language_type=tts_language,
    )
    return stt, llm, tts


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("voice debug: starting bot")
    stt, llm, tts = build_services()
    vad_processor = VADProcessor(vad_analyzer=build_local_vad_analyzer())
    client_frame_filter = ClientFrameCompatibilityFilter()

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[ExternalUserTurnStopStrategy()],
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            vad_processor,
            stt,
            user_aggregator,
            llm,
            tts,
            client_frame_filter,
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

    @user_aggregator.event_handler("on_user_turn_message_added")
    async def on_user_turn_message_added(aggregator, message: UserTurnMessageAddedMessage):
        logger.info("voice debug: user turn message added: {}", message.content)

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        logger.info(
            "voice debug: user turn stopped strategy={} content={}",
            type(strategy).__name__,
            message.content,
        )

    @llm.event_handler("on_completion_timeout")
    async def on_completion_timeout(service):
        logger.warning("voice debug: llm completion timeout")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("voice debug: client connected")
        greeting = env_first("AXION_PIPECAT_GREETING")
        if greeting:
            context.add_message({"role": "developer", "content": greeting})
            await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("voice debug: client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, build_transport_params())
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
