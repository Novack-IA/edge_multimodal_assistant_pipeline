#!/usr/bin/env python3
"""Monta e (opcionalmente) executa o comando `vllm serve` a partir do
`config.yaml` — a quantização/precisão é trocável só editando o config
(FR-12b, docs/13 §5), nunca hardcoded aqui.

Uso:
    python deployment/serve_vllm.py            # imprime o comando
    python deployment/serve_vllm.py --exec      # roda de verdade (exec)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config

# Formatos de quantização servidos diretamente pelo vLLM (docs/13 §5) — GGUF
# é fallback via llama.cpp, não passa por aqui (docs/13 §8).
VLLM_PRECISIONS = {"bf16", "fp8", "nvfp4"}


def build_command(cfg) -> list[str]:
    llm = cfg.llm
    if llm.runtime != "vllm":
        raise SystemExit(
            f"llm.runtime={llm.runtime!r} não é 'vllm' — use deployment/serve_llama_cpp.md (docs/13 §8)."
        )
    if llm.precision not in VLLM_PRECISIONS:
        raise SystemExit(
            f"llm.precision={llm.precision!r} não é servível pelo vLLM diretamente "
            f"(válidos: {sorted(VLLM_PRECISIONS)}); GGUF vai por llama.cpp (docs/13 §8)."
        )

    server_url = llm.server_url.rstrip("/")
    # server_url é tipo http://127.0.0.1:8000/v1 — extrai host:port.
    without_scheme = server_url.split("://", 1)[-1]
    host_port = without_scheme.split("/", 1)[0]
    host, _, port = host_port.partition(":")

    cmd = [
        "vllm",
        "serve",
        llm.backbone,
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        llm.tool_call_parser,
        "--max-model-len",
        str(llm.context_length),
        "--gpu-memory-utilization",
        str(llm.gpu_memory_utilization),
        "--host",
        host or "127.0.0.1",
        "--port",
        port or "8000",
    ]
    if llm.precision != "bf16":
        cmd += ["--quantization", llm.precision]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="caminho alternativo para config.yaml")
    parser.add_argument("--exec", action="store_true", help="executa o comando (os.execvp) em vez de só imprimir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cmd = build_command(cfg)

    print(" \\\n  ".join(cmd))
    if args.exec:
        os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
