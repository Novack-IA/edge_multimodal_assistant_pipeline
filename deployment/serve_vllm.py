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
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import REPO_ROOT, load_config

# Formatos de quantização servidos diretamente pelo vLLM (docs/13 §5) — GGUF
# é fallback via llama.cpp, não passa por aqui (docs/13 §8).
VLLM_PRECISIONS = {"bf16", "fp8", "nvfp4"}


def find_pip_cuda_home(venv_dir: Path) -> Path | None:
    """O `nvidia-cuda-nvcc` do pip (dependência transitiva do vLLM) traz um
    nvcc completo, mas não o registra em PATH/CUDA_HOME — sem isso o JIT do
    FlashInfer falha com "Could not find nvcc" mesmo com a GPU visível
    (visto no device: SM_110a/Thor, vLLM 0.28.0, sem CUDA Toolkit de sistema).
    Localiza `.../nvidia/cu*/bin/nvcc` dentro do venv do vLLM."""
    matches = glob.glob(str(venv_dir / "lib" / "python3.*" / "site-packages" / "nvidia" / "cu*" / "bin" / "nvcc"))
    if not matches:
        return None
    return Path(matches[0]).parent.parent


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

    # vLLM roda em venv isolado do app (dependências pesadas/instáveis não
    # devem clobar o venv do orquestrador — ver deployment/README.md).
    vllm_bin = REPO_ROOT / ".venv-vllm" / "bin" / "vllm"
    cmd = [
        str(vllm_bin) if vllm_bin.exists() else "vllm",
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
        env = os.environ.copy()
        venv_vllm = REPO_ROOT / ".venv-vllm"
        if str(venv_vllm / "bin" / "vllm") == cmd[0]:
            # .venv-vllm/bin na PATH: dá ao JIT do FlashInfer acesso ao `ninja`
            # do pip (subprocess.run(["ninja", ...]) resolve por PATH, não
            # pelo venv do interpretador que o invocou).
            env["PATH"] = f"{venv_vllm / 'bin'}:{env.get('PATH', '')}"
            cuda_home = find_pip_cuda_home(venv_vllm)
            if cuda_home is not None and not env.get("CUDA_HOME"):
                env["CUDA_HOME"] = str(cuda_home)
                env["PATH"] = f"{cuda_home / 'bin'}:{env['PATH']}"
                print(f"# CUDA_HOME={cuda_home} (nvcc do pip nvidia-cuda-nvcc)")
        os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
