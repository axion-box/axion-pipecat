"""QwenASRProcessor - Pipecat FrameProcessor bridging to Alibaba Cloud DashScope realtime ASR.

Protocol: DashScope realtime WebSocket (JSON TextMessage, Base64 PCM16 audio).
Endpoint: wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-asr-flash-realtime
Auth: Authorization: bearer <DASHSCOPE_API_KEY>
"""

import asyncio
import base64
import json
from collections import deque

import websockets
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.time import time_now_iso8601


class QwenASRProcessor(FrameProcessor):
    """Custom ASR processor using Alibaba Cloud DashScope Qwen-ASR realtime API.

    Receives AudioRawFrame from the pipeline, sends base64-encoded PCM16 audio
    as JSON TextMessages over WebSocket to DashScope, and produces
    transcription frames. Turn start is gated by local Pipecat VAD, while
    turn commit is confirmed by DashScope server VAD when enabled.

    Unlike Pipecat's built-in STT services (which extend STTService and receive
    raw PCM bytes), this processor handles the entire WebSocket lifecycle because
    DashScope uses JSON TextMessage protocol, not raw binary audio.
    """

    ENDPOINT = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "qwen3-asr-flash-realtime",
        endpoint: str | None = None,
        sample_rate: int = 16000,
        silence_duration_ms: int = 400,
        server_vad: bool = True,
        preroll_secs: float = 0.4,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint or self.ENDPOINT
        self._sample_rate = sample_rate
        self._silence_duration_ms = silence_duration_ms
        self._server_vad = server_vad
        self._preroll_secs = preroll_secs

        self._ws: websockets.WebSocketClientProtocol | None = None
        self._recv_task: asyncio.Task | None = None
        self._connected = False
        self._started = False
        self._ready_event = asyncio.Event()
        self._audio_chunks_seen = 0
        self._send_audio_enabled = False
        self._local_turn_active = False
        self._server_speech_active = False
        self._preroll_audio: deque[bytes] = deque()
        self._preroll_audio_bytes = 0
        self._preroll_max_bytes = max(1, int(self._sample_rate * self._preroll_secs) * 2)

    # ── FrameProcessor lifecycle ──────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._on_start()
            await self.push_frame(frame, direction)
        elif isinstance(frame, (CancelFrame, EndFrame)):
            await self._on_stop()
            await self.push_frame(frame, direction)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            if self._connected and self._started:
                await self._handle_local_vad_start()
            await self.push_frame(frame, direction)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._connected and self._started:
                logger.debug(f"{self}: local VAD speech stopped; waiting for server commit")
            await self.push_frame(frame, direction)
        elif isinstance(frame, AudioRawFrame):
            if self._connected and self._started:
                self._audio_chunks_seen += 1
                if self._audio_chunks_seen <= 5 or self._audio_chunks_seen % 50 == 0:
                    logger.debug(
                        f"{self}: audio chunk #{self._audio_chunks_seen} bytes={len(frame.audio)}"
                    )
                self._append_preroll_audio(frame.audio)
                if self._send_audio_enabled:
                    await self._send_audio(frame.audio)
            # Audio input is consumed by ASR and should not continue
            # downstream to the LLM/TTS/output transport chain.
        else:
            await self.push_frame(frame, direction)

    async def _on_start(self):
        self._started = True
        self.create_task(self._connect())

    async def _on_stop(self):
        self._started = False
        self._ready_event.clear()
        self._reset_turn_state(clear_preroll=True)
        if self._ws:
            try:
                await self._send_event({"type": "session.finish"})
            except Exception:
                pass
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._recv_task:
            await self.cancel_task(self._recv_task)

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def wait_until_ready(self, timeout: float = 10.0) -> bool:
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    # ── WebSocket connection management ──────────────────────────────

    async def _connect(self):
        url = f"{self._endpoint}?model={self._model}"
        headers = {"Authorization": f"bearer {self._api_key}"}

        try:
            self._ws = await websockets.connect(url, additional_headers=headers)
            self._connected = True
            self._ready_event.set()
            logger.info(f"{self}: connected to DashScope ASR ({self._model})")
            self._recv_task = self.create_task(self._recv_loop())
            # Send session.update to configure
            session = {
                "input_audio_format": "pcm",
                "sample_rate": self._sample_rate,
                "channels": 1,
            }
            if self._server_vad:
                session["turn_detection"] = {
                    "type": "server_vad",
                    "silence_duration_ms": self._silence_duration_ms,
                }
            else:
                session["turn_detection"] = None
            await self._send_event({
                "type": "session.update",
                "session": session,
            })
        except Exception as e:
            logger.error(f"{self}: failed to connect: {e}")
            self._ready_event.clear()

    async def _send_event(self, event: dict):
        if self._ws:
            await self._ws.send(json.dumps(event, ensure_ascii=False))

    async def _send_audio(self, audio: bytes):
        """Send PCM16 audio as base64 JSON."""
        if not self._ws or not audio:
            return
        audio_b64 = base64.b64encode(audio).decode("ascii")
        await self._send_event({
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        })

    def _append_preroll_audio(self, audio: bytes):
        if not audio:
            return

        self._preroll_audio.append(audio)
        self._preroll_audio_bytes += len(audio)

        while self._preroll_audio and self._preroll_audio_bytes > self._preroll_max_bytes:
            dropped = self._preroll_audio.popleft()
            self._preroll_audio_bytes -= len(dropped)

    def _clear_preroll_audio(self):
        self._preroll_audio.clear()
        self._preroll_audio_bytes = 0

    def _reset_turn_state(self, *, clear_preroll: bool):
        self._send_audio_enabled = False
        self._local_turn_active = False
        self._server_speech_active = False
        if clear_preroll:
            self._clear_preroll_audio()

    async def _handle_local_vad_start(self):
        self._local_turn_active = True
        if self._send_audio_enabled:
            logger.debug(f"{self}: local VAD speech started while ASR upload already active")
            return

        self._send_audio_enabled = True
        preroll_chunks = list(self._preroll_audio)
        preroll_bytes = self._preroll_audio_bytes
        logger.debug(
            f"{self}: local VAD speech started; enabling ASR upload with preroll_bytes={preroll_bytes}"
        )
        for chunk in preroll_chunks:
            await self._send_audio(chunk)

    async def _handle_server_speech_started(self, event: dict):
        self._server_speech_active = True
        logger.debug(f"{self}: server VAD speech started: {event}")

    async def _handle_server_speech_stopped(self, event: dict):
        self._server_speech_active = False
        logger.debug(f"{self}: server VAD speech stopped: {event}")

    async def _handle_server_committed(self, event: dict):
        logger.debug(f"{self}: server committed turn: {event}")
        should_emit_stop = self._local_turn_active or self._send_audio_enabled or self._server_speech_active
        self._reset_turn_state(clear_preroll=True)
        if should_emit_stop:
            await self.broadcast_frame(UserStoppedSpeakingFrame)

    # ── Receive loop ──────────────────────────────────────────────────

    async def _recv_loop(self):
        try:
            async for raw in self._ws:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                event_type = event.get("type", "")

                if event_type == "conversation.item.input_audio_transcription.text":
                    text = event.get("text", "") + event.get("stash", "")
                    if text.strip():
                        logger.debug(f"{self}: interim: {text.strip()}")
                        await self.push_frame(
                            InterimTranscriptionFrame(
                                text=text.strip(),
                                user_id="user",
                                timestamp=time_now_iso8601(),
                                result=event,
                            )
                        )

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    text = event.get("transcript", "")
                    if text.strip():
                        logger.info(f"{self}: final: {text.strip()}")
                        await self.push_frame(
                            TranscriptionFrame(
                                text=text.strip(),
                                user_id="user",
                                timestamp=time_now_iso8601(),
                                result=event,
                                finalized=True,
                            )
                        )

                elif event_type == "input_audio_buffer.speech_started":
                    await self._handle_server_speech_started(event)

                elif event_type == "input_audio_buffer.speech_stopped":
                    await self._handle_server_speech_stopped(event)

                elif event_type == "input_audio_buffer.committed":
                    await self._handle_server_committed(event)

                elif event_type == "session.finished":
                    logger.info(f"{self}: session finished")
                    break

                elif event_type == "error":
                    logger.error(f"{self}: ASR error: {event}")

        except websockets.ConnectionClosed:
            logger.warning(f"{self}: connection closed")
        except Exception as e:
            logger.error(f"{self}: recv error: {e}")
        finally:
            self._connected = False
            self._ready_event.clear()
            self._ws = None
            self._reset_turn_state(clear_preroll=True)
