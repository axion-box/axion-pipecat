"""QwenTTSProcessor - Pipecat FrameProcessor bridging to Alibaba Cloud DashScope realtime TTS.

Protocol: DashScope realtime WebSocket (JSON TextMessage for text input,
         JSON TextMessage + binary audio deltas for output).
Endpoint: wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-tts-flash-realtime
Auth: Authorization: bearer <DASHSCOPE_API_KEY>
"""

import asyncio
import base64
import json
import re
from collections import deque

import websockets
from loguru import logger

from pipecat.frames.frames import (
    AggregationType,
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    StartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class QwenTTSProcessor(FrameProcessor):
    """Custom TTS processor using Alibaba Cloud DashScope Qwen-TTS realtime API.

    Receives TextFrame, sends it as JSON over WebSocket to DashScope TTS,
    receives audio delta events (base64 PCM), and produces TTSAudioRawFrame
    downstream.

    Unlike Pipecat's built-in TTSService, this handles the full DashScope
    protocol including session management, text buffering, and commit.
    """

    ENDPOINT = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "qwen3-tts-flash-realtime",
        endpoint: str | None = None,
        voice: str = "Chelsie",  # DashScope TTS voice
        sample_rate: int = 24000,
        language_type: str = "Chinese",
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint or self.ENDPOINT
        self._voice = voice
        self._sample_rate = 24000
        if sample_rate != 24000:
            logger.warning(
                f"{self}: Qwen-TTS-Realtime only supports 24000 Hz output; "
                f"requested {sample_rate} Hz will be ignored"
            )
        self._language_type = language_type

        self._ws: websockets.WebSocketClientProtocol | None = None
        self._recv_task: asyncio.Task | None = None
        self._connected = False
        self._started = False
        self._ready_event = asyncio.Event()
        self._utterance_index = 0
        self._pending_utterances: deque[dict] = deque()
        self._pending_response_end_frame: LLMFullResponseEndFrame | None = None
        self._text_buffer = ""
        self._text_buffer_includes_inter_frame_spaces = False
        self._text_buffer_append_to_context = True

    # ── FrameProcessor lifecycle ──────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._on_start()
            await self.push_frame(frame, direction)
        elif isinstance(frame, CancelFrame):
            await self._on_stop()
            await self.push_frame(frame, direction)
        elif isinstance(frame, EndFrame):
            await self._flush_text_buffer()
            await self._on_stop()
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            await self._on_interruption()
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._flush_text_buffer()
            if self._pending_utterances:
                self._pending_response_end_frame = frame
            else:
                await self.push_frame(frame, direction)
        elif isinstance(frame, TextFrame):
            if self._connected and self._started and frame.text:
                self._buffer_text(frame)
                if self._should_flush_text_buffer():
                    await self._flush_text_buffer()
        else:
            await self.push_frame(frame, direction)

    async def _on_start(self):
        self._started = True
        self.create_task(self._connect())

    async def _on_stop(self):
        self._started = False
        self._ready_event.clear()
        self._pending_utterances.clear()
        self._pending_response_end_frame = None
        self._reset_text_buffer()
        if self._ws:
            try:
                await self._send_event({"type": "session.finish"})
            except Exception:
                pass
        if self._recv_task:
            try:
                await self.cancel_task(self._recv_task)
            except asyncio.CancelledError:
                pass

    async def _on_interruption(self):
        """Clear pending local state on interruption.

        DashScope's realtime Qwen TTS endpoint rejects
        ``input_text_buffer.clear`` unless the session is using a compatible
        client-commit mode, so we only reset local buffers here.
        """
        self._pending_utterances.clear()
        self._pending_response_end_frame = None
        self._reset_text_buffer()

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
            logger.info(f"{self}: connected to DashScope TTS ({self._model})")
            self._recv_task = self.create_task(self._recv_loop())
            # Configure session
            await self._send_event({
                "type": "session.update",
                "session": {
                    "voice": self._voice,
                    "response_format": "pcm",
                    "sample_rate": self._sample_rate,
                    "language_type": self._language_type,
                    "channels": 1,
                },
            })
        except Exception as e:
            logger.error(f"{self}: failed to connect TTS: {e}")
            self._ready_event.clear()

    async def _send_event(self, event: dict):
        if self._ws:
            await self._ws.send(json.dumps(event, ensure_ascii=False))

    def _next_context_id(self) -> str:
        self._utterance_index += 1
        return f"tts-turn-{self._utterance_index}"

    def _reset_text_buffer(self):
        self._text_buffer = ""
        self._text_buffer_includes_inter_frame_spaces = False
        self._text_buffer_append_to_context = True

    def _buffer_text(self, frame: TextFrame):
        self._text_buffer += frame.text
        self._text_buffer_includes_inter_frame_spaces = (
            self._text_buffer_includes_inter_frame_spaces or frame.includes_inter_frame_spaces
        )
        self._text_buffer_append_to_context = (
            self._text_buffer_append_to_context and frame.append_to_context
        )

    def _should_flush_text_buffer(self) -> bool:
        text = self._text_buffer.strip()
        if not text:
            return False

        if re.search(r"[。！？!?]\s*$", text):
            return True

        if len(text) >= 60 and re.search(r"[，,；;：:\n]\s*$", text):
            return True

        return len(text) >= 120

    async def _flush_text_buffer(self):
        text = self._text_buffer.strip()
        if not text:
            self._reset_text_buffer()
            return

        await self._push_text(
            text=text,
            includes_inter_frame_spaces=self._text_buffer_includes_inter_frame_spaces,
            append_to_context=self._text_buffer_append_to_context,
        )
        self._reset_text_buffer()

    async def _push_text(
        self,
        *,
        text: str,
        includes_inter_frame_spaces: bool,
        append_to_context: bool,
    ):
        """Send text to TTS, then commit to trigger synthesis."""
        if not self._ws:
            return
        context_id = self._next_context_id()
        self._pending_utterances.append(
            {
                "context_id": context_id,
                "text": text,
                "includes_inter_frame_spaces": includes_inter_frame_spaces,
                "append_to_context": append_to_context,
            }
        )
        await self._send_event({
            "type": "input_text_buffer.append",
            "text": text,
        })
        await self._send_event({
            "type": "input_text_buffer.commit",
        })
        await self.push_frame(TTSStartedFrame(context_id=context_id))

    # ── Receive loop ──────────────────────────────────────────────────

    async def _recv_loop(self):
        try:
            async for raw in self._ws:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                event_type = event.get("type", "")

                if event_type == "response.audio.delta":
                    # Audio is base64-encoded PCM in the "delta" field
                    delta_b64 = event.get("delta", "")
                    if delta_b64:
                        try:
                            audio_bytes = base64.b64decode(delta_b64)
                        except Exception:
                            continue
                        await self.push_frame(TTSAudioRawFrame(
                            audio=audio_bytes,
                            sample_rate=self._sample_rate,
                            num_channels=1,
                            context_id=self._pending_utterances[0]["context_id"]
                            if self._pending_utterances
                            else None,
                        ))

                elif event_type == "response.audio.done":
                    logger.debug(f"{self}: audio done")
                    utterance = self._pending_utterances.popleft() if self._pending_utterances else None
                    if utterance:
                        text_frame = TTSTextFrame(
                            utterance["text"],
                            AggregationType.SENTENCE,
                            context_id=utterance["context_id"],
                        )
                        text_frame.will_be_spoken = True
                        text_frame.includes_inter_frame_spaces = utterance[
                            "includes_inter_frame_spaces"
                        ]
                        text_frame.append_to_context = utterance["append_to_context"]
                        await self.push_frame(text_frame)
                        await self.push_frame(TTSStoppedFrame(context_id=utterance["context_id"]))
                    else:
                        await self.push_frame(TTSStoppedFrame())

                    if self._pending_response_end_frame and not self._pending_utterances:
                        frame = self._pending_response_end_frame
                        self._pending_response_end_frame = None
                        await self.push_frame(frame)

                elif event_type == "session.finished":
                    logger.info(f"{self}: TTS session finished")
                    break

                elif event_type == "error":
                    logger.error(f"{self}: TTS error: {event}")

        except websockets.ConnectionClosed:
            logger.warning(f"{self}: TTS connection closed")
        except Exception as e:
            logger.error(f"{self}: TTS recv error: {e}")
        finally:
            self._connected = False
            self._ready_event.clear()
