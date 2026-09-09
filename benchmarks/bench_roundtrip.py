#!/usr/bin/env python3
"""NFR-L6 — round-trip completo, com STT/LLM/TTS **reais** (nada mockado):
Parakeet-TDT (services/stt), vLLM servindo LFM2.5-VL-3B de verdade
(services/llm), XTTS-v2 (services/tts). Mede a cascata inteira,
fronteira a fronteira (CLAUDE.md "Logging estruturado"):

  fala do usuário (simulada via TTS) ──▶ STT final ──▶ LLM 1º token
       ──▶ LLM 1ª frase completa ──▶ TTS 1º chunk de áudio

Metodologia: docs/02 §2 (NFR-L1..L8) e docs/11 §3 (p50/p95, nunca só
média). Benchmarks de referência citados no relatório desta sessão —
Stivers et al. 2009 (PNAS, gap de turno humano), ITU-T G.114, e práticas
de produção 2026 (LiveKit/Vapi/OpenAI Realtime) — ver docs/02 e a
conversa que motivou este script para as fontes completas.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np

from benchmarks.harness import LatencyBench
from config import load_config
from contracts.messages import LlmToken, SamplingCfg
from orchestrator.pipeline import is_speakable_chunk_boundary
from services.llm.client import LlmClient
from services.stt import create_stt_engine
from services.tts.worker import XttsV2Engine

SYSTEM_PROMPT = "Você é um assistente de voz em pt-BR. Responda em UMA frase curta e natural."

PERGUNTAS = [
    "Qual é a capital do Brasil?",
    "Me dá uma dica rápida de produtividade.",
    "Qual é a diferença entre RAM e armazenamento?",
    "Como faço um café coado?",
    "O que é inteligência artificial, em uma frase?",
    "Qual é a distância aproximada da Terra até a Lua?",
    "Me sugere um filme de ficção científica.",
    "Por que o céu é azul?",
]


async def _synthesize_user_audio(tts: XttsV2Engine, text: str) -> np.ndarray:
    """Usa o próprio XTTS para simular a fala do usuário (16kHz, entrada
    esperada pelo STT) — evita depender de um dataset de voz gravado."""
    cancel = asyncio.Event()
    chunks = [out.pcm async for out in tts.synthesize_stream(text, cancel) if out.pcm]
    pcm = b"".join(chunks)
    arr24k = np.frombuffer(pcm, dtype=np.float32)
    return librosa.resample(arr24k, orig_sr=24000, target_sr=16000)


async def main(n: int = 8) -> None:
    cfg = load_config()
    stt = create_stt_engine(cfg.stt)
    llm = LlmClient(cfg.llm)
    tts = XttsV2Engine(cfg.tts.model_copy(update={"use_deepspeed": False}))

    if not await llm.health_check():
        print(f"vLLM não está respondendo em {cfg.llm.server_url} — suba o servidor primeiro.")
        raise SystemExit(1)

    print("Aquecendo TTS (carrega XTTS-v2 + extrai latents da voz de referência)...")
    await tts.warmup()
    print("Aquecendo LLM (turno dummy pós-boot, NFR-R6)...")
    await llm.warmup()
    print("Aquecendo STT (1a chamada paga JIT/cuDNN algo-search — não deve contar na medição)...")
    dummy_audio = np.zeros(16000, dtype=np.float32)
    await stt.transcribe(dummy_audio, partial=False)

    perguntas = PERGUNTAS[:n]
    print(f"Sintetizando {len(perguntas)} falas de usuário simuladas (fora do trecho medido)...")
    audios = [(p, await _synthesize_user_audio(tts, p)) for p in perguntas]

    bench_stt = LatencyBench("stt_final (Parakeet)")
    bench_ttft = LatencyBench("llm_ttft (vLLM, 1o token)")
    bench_first_sentence = LatencyBench("llm_first_speakable_chunk (1o trecho falável)")
    bench_ttfb = LatencyBench("tts_ttfb (XTTS-v2, 1o chunk)")
    bench_roundtrip = LatencyBench("roundtrip_e2e (fim da fala -> 1o audio)")

    for pergunta, audio in audios:
        t0 = time.perf_counter()

        transcript = await stt.transcribe(audio, partial=False)
        t1 = time.perf_counter()
        bench_stt.samples_ms.append((t1 - t0) * 1000)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript.text},
        ]
        cancel = asyncio.Event()
        buffer = ""
        t_first_token: float | None = None
        t_first_sentence: float | None = None
        async for event in llm.stream_chat(messages, tools=[], sampling=SamplingCfg(max_new_tokens=60), cancel_token=cancel):
            if isinstance(event, LlmToken) and event.text:
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                buffer += event.text
                if is_speakable_chunk_boundary(buffer):
                    t_first_sentence = time.perf_counter()
                    cancel.set()  # já medimos — libera o vLLM de continuar gerando o resto
                    break
        t_first_sentence = t_first_sentence or time.perf_counter()
        t_first_token = t_first_token or t_first_sentence
        bench_ttft.samples_ms.append((t_first_token - t1) * 1000)
        bench_first_sentence.samples_ms.append((t_first_sentence - t1) * 1000)

        cancel2 = asyncio.Event()
        t_first_chunk: float | None = None
        async for out in tts.synthesize_stream(buffer.strip() or "Certo.", cancel2):
            if out.pcm:
                t_first_chunk = time.perf_counter()
                cancel2.set()  # já medimos TTFB — sinaliza a thread produtora a parar
                break
        t_end = t_first_chunk or time.perf_counter()
        bench_ttfb.samples_ms.append((t_end - t_first_sentence) * 1000)
        bench_roundtrip.samples_ms.append((t_end - t0) * 1000)

        print(f"  P: {pergunta!r}\n  T(stt): {transcript.text!r}\n  R(llm): {buffer.strip()!r}"
              f"\n  roundtrip: {(t_end - t0) * 1000:.0f}ms\n")

    print("=" * 70)
    for b in (bench_stt, bench_ttft, bench_first_sentence, bench_ttfb, bench_roundtrip):
        print(b.report())
    print("=" * 70)

    budgets = cfg.latency_budgets_ms
    checks = [
        (bench_ttft, {"p50_ms": budgets.llm_ttft_p50}),
        (bench_ttfb, {"p50_ms": budgets.tts_ttfb_p50}),
        (bench_roundtrip, {"p50_ms": budgets.roundtrip_p50, "p95_ms": budgets.roundtrip_p95}),
    ]
    all_ok = True
    for bench, kwargs in checks:
        try:
            bench.assert_within_budget(**kwargs)
            print(f"OK   {bench.name}")
        except AssertionError as exc:
            all_ok = False
            print(f"FALHOU {exc}")
    print("Todos os orçamentos respeitados." if all_ok else "Nem todo orçamento foi respeitado — ver acima.")

    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
