#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time


PIPECAT_ROOT = pathlib.Path(__file__).resolve().parent
ROOT = PIPECAT_ROOT.parent.parent
ENV_PATH = ROOT / ".env"
SCRIPT_LABEL = pathlib.Path(__file__).name

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
DEFAULT_CLIENT_PORT = 5173
DEFAULT_TRANSPORT_OFFICIAL = "webrtc"
DEFAULT_TRANSPORT_DEMO = "websocket"
DEFAULT_ZHIZENGZENG_BASE_URL = "https://api.zhizengzeng.com/v1"
MODE_OFFICIAL = "official"
MODE_DEMO = "demo"
VOICE_SCRIPTS = {
    MODE_OFFICIAL: PIPECAT_ROOT / "examples" / "axion_official_voice_debug" / "server.py",
    MODE_DEMO: PIPECAT_ROOT / "examples" / "axion_glenclaw_voice_debug" / "server.py",
}
DEMO_CLIENT_DIR = PIPECAT_ROOT / "examples" / "axion_glenclaw_voice_debug" / "client"


def log(message: str) -> None:
    print(f"[{SCRIPT_LABEL}] {message}", flush=True)


def fail(message: str) -> SystemExit:
    return SystemExit(f"{SCRIPT_LABEL}: {message}")


def strings_trim(value: str | None) -> str:
    return value.strip() if value is not None else ""


def strip_wrapping_quotes(value: str) -> str:
    trimmed = value.strip()
    if len(trimmed) >= 2 and trimmed[0] == trimmed[-1] and trimmed[0] in {'"', "'"}:
        return trimmed[1:-1].strip()
    return trimmed


def read_env_file(path: pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = strip_wrapping_quotes(value)
    return values


def read_script_env_value(root_env: dict[str, str], key: str) -> str:
    process_value = strings_trim(os.environ.get(key, ""))
    if process_value:
        return process_value
    return strings_trim(root_env.get(key, ""))


def is_zhizengzeng_base_url(value: str) -> bool:
    return "api.zhizengzeng.com" in value.strip().lower()


def resolve_pipecat_python() -> pathlib.Path:
    python_bin = PIPECAT_ROOT / ".venv" / "bin" / "python"
    if not python_bin.exists():
        raise fail(
            f"missing Pipecat venv python: {python_bin}. "
            "Set up thirdparty/pipecat/.venv first."
        )
    return python_bin


def resolve_transport(mode: str, requested: str) -> str:
    trimmed = strings_trim(requested)
    if trimmed:
        return trimmed
    if mode == MODE_DEMO:
        return DEFAULT_TRANSPORT_DEMO
    return DEFAULT_TRANSPORT_OFFICIAL


def terminate_process(process: subprocess.Popen[bytes | str] | None, label: str) -> None:
    if process is None or process.poll() is not None:
        return

    log(f"stopping {label} (pid={process.pid})")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def build_pipecat_env(
    root_env: dict[str, str],
    *,
    mode: str,
    llm_api_key_override: str,
    llm_base_url_override: str,
    llm_model_override: str,
) -> dict[str, str]:
    process_env = os.environ.copy()

    llm_base_url = strings_trim(llm_base_url_override) or read_script_env_value(
        root_env, "AXION_PIPECAT_LLM_BASE_URL"
    )
    if (
        mode == MODE_OFFICIAL
        and not llm_base_url
        and strings_trim(os.environ.get("ZHIZENGZENG_API_KEY", ""))
    ):
        llm_base_url = DEFAULT_ZHIZENGZENG_BASE_URL
    if (
        mode == MODE_OFFICIAL
        and not llm_base_url
        and strings_trim(os.environ.get("ZHIZENZENG_API_KEY", ""))
    ):
        llm_base_url = DEFAULT_ZHIZENGZENG_BASE_URL
    if llm_base_url:
        process_env["AXION_PIPECAT_LLM_BASE_URL"] = llm_base_url

    llm_api_key = strings_trim(llm_api_key_override)
    llm_api_key_source = "--llm-api-key"
    if not llm_api_key:
        llm_api_key_candidates = ["AXION_PIPECAT_LLM_API_KEY"]
        if is_zhizengzeng_base_url(llm_base_url) or mode == MODE_OFFICIAL:
            llm_api_key_candidates.extend(["ZHIZENGZENG_API_KEY", "ZHIZENZENG_API_KEY"])
        llm_api_key_candidates.extend(["OPENAI_API_KEY", "DASHSCOPE_API_KEY", "DASHSCOPE_SECRET_KEY"])
        for candidate in llm_api_key_candidates:
            llm_api_key = read_script_env_value(root_env, candidate)
            if llm_api_key:
                llm_api_key_source = candidate
                break
    if llm_api_key:
        process_env["AXION_PIPECAT_LLM_API_KEY"] = llm_api_key
        process_env["OPENAI_API_KEY"] = llm_api_key
        process_env["QWEN_API_KEY"] = llm_api_key
        if mode == MODE_OFFICIAL or is_zhizengzeng_base_url(llm_base_url):
            process_env["ZHIZENGZENG_API_KEY"] = llm_api_key
            process_env["ZHIZENZENG_API_KEY"] = llm_api_key
            if llm_api_key_source == "OPENAI_API_KEY":
                log(
                    "warning: official voice debug is using OPENAI_API_KEY against "
                    "api.zhizengzeng.com; prefer AXION_PIPECAT_LLM_API_KEY or "
                    "ZHIZENGZENG_API_KEY to avoid key mix-ups."
                )

    llm_model = strings_trim(llm_model_override) or read_script_env_value(
        root_env, "AXION_PIPECAT_LLM_MODEL"
    )
    if llm_model:
        process_env["AXION_PIPECAT_LLM_MODEL"] = llm_model

    return process_env


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pipecat voice debug demos.")
    parser.add_argument(
        "mode",
        choices=[MODE_OFFICIAL, MODE_DEMO],
        help="official: OpenAI-compatible STT/LLM/TTS path; demo: Glenclaw LLM + Qwen ASR/TTS demo",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="HTTP host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP port")
    parser.add_argument("--transport", default="", help="transport type")
    parser.add_argument(
        "--client-port",
        type=int,
        default=DEFAULT_CLIENT_PORT,
        help="custom demo client port (demo mode only)",
    )
    parser.add_argument("--llm-base-url", default="", help="override LLM base URL")
    parser.add_argument("--llm-model", default="", help="override LLM model")
    parser.add_argument(
        "--llm-api-key",
        "--openai-api-key",
        dest="llm_api_key",
        default="",
        help="override LLM API key",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args(sys.argv[1:])
    script_path = VOICE_SCRIPTS[args.mode]
    if not script_path.exists():
        raise fail(f"missing Pipecat script: {script_path}")

    root_env = read_env_file(ENV_PATH)
    env = build_pipecat_env(
        root_env,
        mode=args.mode,
        llm_api_key_override=args.llm_api_key,
        llm_base_url_override=args.llm_base_url,
        llm_model_override=args.llm_model,
    )
    transport = resolve_transport(args.mode, args.transport)

    python_bin = resolve_pipecat_python()
    command = [
        str(python_bin),
        str(script_path.relative_to(PIPECAT_ROOT)),
        "-t",
        transport,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    log(f"mode: {args.mode}")
    if args.mode == MODE_DEMO and transport == "websocket":
        client_url = f"http://{args.host}:{args.client_port}"
        log(f"url: {client_url}")
        log(f"bot start: http://{args.host}:{args.port}/start")
    else:
        log(f"url: http://{args.host}:{args.port}")
    log(f"exec: {' '.join(command)}")

    if args.mode == MODE_DEMO and transport == "websocket":
        if not DEMO_CLIENT_DIR.exists():
            raise fail(f"missing demo client dir: {DEMO_CLIENT_DIR}")
        if not (DEMO_CLIENT_DIR / "node_modules").exists():
            raise fail(
                f"missing client node_modules in {DEMO_CLIENT_DIR}. "
                "Run `npm install` there first."
            )

        client_env = os.environ.copy()
        client_env["VITE_BOT_START_URL"] = f"http://{args.host}:{args.port}/start"
        client_command = [
            "npm",
            "run",
            "dev",
            "--",
            "--host",
            args.host,
            "--port",
            str(args.client_port),
        ]
        log(f"client: {' '.join(client_command)}")

        bot_process: subprocess.Popen[str] | None = None
        client_process: subprocess.Popen[str] | None = None
        try:
            bot_process = subprocess.Popen(command, cwd=PIPECAT_ROOT, env=env, text=True)
            client_process = subprocess.Popen(
                client_command,
                cwd=DEMO_CLIENT_DIR,
                env=client_env,
                text=True,
            )
            while True:
                bot_returncode = bot_process.poll()
                client_returncode = client_process.poll()
                if bot_returncode is not None:
                    raise subprocess.CalledProcessError(bot_returncode, command)
                if client_returncode is not None:
                    raise subprocess.CalledProcessError(client_returncode, client_command)
                time.sleep(0.5)
        except KeyboardInterrupt:
            log("received interrupt, shutting down")
        finally:
            terminate_process(client_process, "demo client")
            terminate_process(bot_process, "demo bot")
        return 0

    subprocess.run(command, cwd=PIPECAT_ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
