# Axion Glenclaw Voice Debug Demo

This demo is for debugging a plain voice conversation loop in Pipecat:

- browser microphone
- Qwen realtime ASR
- Glenclaw OpenAI-compatible LLM
- Qwen realtime TTS

It intentionally does **not** call the Axion agent runtime in the middle.

## What uses Glenclaw

- LLM: defaults to `GLENCLAW_BASE_URL + /api/v1/openai`
- API key: `GLENCLAW_CLOUD_SK`

## What uses speech credentials

By default ASR/TTS read one of:

- `AXION_PIPECAT_SPEECH_API_KEY`
- `GLENCLAW_SPEECH_API_KEY`
- `DASHSCOPE_SECRET_KEY`
- `DASHSCOPE_API_KEY`

ASR/TTS websocket endpoints default to DashScope realtime. If you have Glenclaw speech websocket
gateways, override them with:

- `AXION_PIPECAT_ASR_WS_URL`
- `AXION_PIPECAT_TTS_WS_URL`

## Run the bot server

From `thirdparty/pipecat`:

```bash
uv run python examples/axion_glenclaw_voice_debug/server.py -t websocket --host 127.0.0.1 --port 7860
```

Optional useful overrides:

```bash
export AXION_PIPECAT_GREETING="Say a short hello when the user connects."
export AXION_PIPECAT_LLM_MODEL="chat"
export AXION_PIPECAT_TTS_VOICE="Cherry"
```

## Run the debug client

From `thirdparty/pipecat/examples/axion_glenclaw_voice_debug/client`:

```bash
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Open:

- [http://127.0.0.1:5173](http://127.0.0.1:5173)

If the embedded browser denies microphone access, open the same URL in your
system browser and allow mic access for `127.0.0.1`.

The UI uses Pipecat's standard `voice-ui-kit` components:

- `PipecatAppBase`
- `UserAudioControl`
- `ConnectButton`
- a custom conversation-first debugger layout with a collapsible `EventsPanel`

## Env for the client

`client/.env.example`:

```bash
VITE_BOT_START_URL="http://127.0.0.1:7860/start"
```

## Notes

- This demo is for local debugging, not the formal Axion runtime path.
- The main UI is conversation-first; raw RTVI events live under the collapsible debug section.
- The demo client uses Pipecat's websocket transport instead of SmallWebRTC so local debugging and custom UI iteration stay simple.
- The client normalizes Pipecat's local `wss://127.0.0.1/...` start response to `ws://127.0.0.1/...` so local debugging works without TLS.
- `python thirdparty/pipecat/debug.py demo` now defaults to the websocket bot plus this custom client on port `5173`.
