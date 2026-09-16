#!/usr/bin/env python3
"""CLI de chat só com o LLM (LFM2.5-VL-3B via vLLM) — sem STT/TTS/câmera.

Pra quando não há acesso a mic/câmera/desktop (ex.: só terminal via SSH) —
usa exatamente os mesmos services/llm/* e tools/* reais do resto do
projeto (Session, LlmClient, run_turn com tool calling), só sem a parte de
voz/visão de demo/streamlit_app.py.

Uso:
    .venv/bin/python demo/cli_chat.py

Comandos:
    /sair     encerra
    /limpar   reseta o histórico da conversa (docs/09)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from services.llm.client import LlmClient
from services.llm.generation_loop import run_turn
from services.llm.session import Session
from tools.builtin import register_builtin_tools
from tools.registry import ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "Você é um assistente em pt-BR rodando localmente numa Jetson AGX Thor. "
    "Responda de forma natural, curta e direta."
)


async def main() -> None:
    cfg = load_config()
    llm = LlmClient(cfg.llm)

    print(f"Conectando ao vLLM em {cfg.llm.server_url} ...")
    if not await llm.health_check():
        print(
            "vLLM não respondeu. Suba com:\n"
            "  .venv/bin/python deployment/serve_vllm.py --exec"
        )
        await llm.aclose()
        return

    print("Aquecendo (turno dummy pós-boot, NFR-R6)...")
    await llm.warmup()
    print(f"Pronto — modelo: {cfg.llm.backbone}\n")
    print("Digite sua mensagem (/sair encerra, /limpar reseta o histórico)\n")

    registry = ToolRegistry(cfg.tools)
    register_builtin_tools(registry)
    session = Session(system_prompt=DEFAULT_SYSTEM_PROMPT, cfg=cfg.llm)

    while True:
        try:
            user_text = input("você> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text == "/sair":
            break
        if user_text == "/limpar":
            session.history.clear()
            print("(histórico limpo)\n")
            continue

        print("assistente> ", end="", flush=True)
        cancel = asyncio.Event()
        async for text in run_turn(session, llm, user_text, [], registry.specs(), registry, cancel):
            print(text, end="", flush=True)
        print("\n")

    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
