"""Demo interativo: conversar de verdade com o assistente.

Reaproveita os mesmos serviços do orquestrador (services/, tools/) — não
há lógica paralela aqui. NÃO é o laço de barge-in completo (docs/08):
Streamlit não sustenta áudio full-duplex em tempo real, então isto é um
teste turn-based (grava/digita → processa → toca a resposta) dos mesmos
componentes reais já validados em benchmarks/bench_roundtrip.py:

    STT (Parakeet-TDT) → LLM (LFM2.5-VL-3B via vLLM) → TTS (XTTS-v2)

Suporta também uma imagem por turno (câmera) — o mesmo contrato de
docs/05 §3 (frame anexado só na mensagem do turno atual, nunca no
histórico persistido).

Uso:
    .venv/bin/streamlit run demo/streamlit_app.py --server.address 0.0.0.0
"""

from __future__ import annotations

import asyncio
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
from PIL import Image

from config import load_config
from contracts.messages import VideoFrame
from services.llm.client import LlmClient
from services.llm.generation_loop import run_turn
from services.llm.session import Session
from services.stt import create_stt_engine
from services.tts.worker import XttsV2Engine
from tools.builtin import register_builtin_tools
from tools.registry import ToolRegistry

SYSTEM_PROMPT = (
    "Você é um assistente de voz em pt-BR, rodando 100% localmente numa Jetson AGX Thor. "
    "Responda de forma natural, curta e direta, como numa conversa falada."
)
TTS_SAMPLE_RATE = 24000

st.set_page_config(page_title="Assistente Multimodal", page_icon="🎙️", layout="centered")


@st.cache_resource(show_spinner=False)
def load_engines():
    cfg = load_config()
    stt = create_stt_engine(cfg.stt)
    llm = LlmClient(cfg.llm)
    tts = XttsV2Engine(cfg.tts.model_copy(update={"use_deepspeed": False}))
    registry = ToolRegistry(cfg.tools)
    register_builtin_tools(registry)
    return cfg, stt, llm, tts, registry


async def _warmup(stt, llm, tts) -> bool:
    if not await llm.health_check():
        return False
    await llm.warmup()
    await tts.warmup()
    await stt.transcribe(np.zeros(16000, dtype=np.float32), partial=False)
    return True


def _decode_audio_to_16k(raw_bytes: bytes) -> np.ndarray:
    data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != 16000:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=16000)
    return mono


def _pcm_f32_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, pcm, sample_rate, format="WAV", subtype="FLOAT")
    return buf.getvalue()


async def _transcribe(stt, pcm16k):
    return await stt.transcribe(pcm16k, partial=False)


async def _respond(session, llm, user_text, frames, registry):
    chunks = []
    async for text in run_turn(session, llm, user_text, frames, registry.specs(), registry, asyncio.Event()):
        chunks.append(text)
    return "".join(chunks)


async def _speak(tts, text):
    pcm_chunks = []
    cancel = asyncio.Event()
    async for out in tts.synthesize_stream(text, cancel):
        if out.pcm:
            pcm_chunks.append(out.pcm)
    return b"".join(pcm_chunks)


cfg, stt, llm, tts, registry = load_engines()

if "session" not in st.session_state:
    st.session_state.session = Session(system_prompt=SYSTEM_PROMPT, cfg=cfg.llm)
if "turns" not in st.session_state:
    st.session_state.turns = []
if "ready" not in st.session_state:
    st.session_state.ready = False

st.title("🎙️ Fale com o LFM2.5-VL-3B")
st.caption(
    f"STT: **{cfg.stt.engine}** · LLM: **{cfg.llm.backbone}** (vLLM, `{cfg.llm.server_url}`) · "
    f"TTS: **{cfg.tts.engine}** — tudo rodando local nesta Jetson AGX Thor. "
    f"Histórico: últimos {cfg.llm.history_max_turns} turnos (docs/09 §6)."
)

if not st.session_state.ready:
    with st.spinner("Aquecendo motores (só na 1ª vez — JIT CUDA, carrega os pesos)..."):
        ok = asyncio.run(_warmup(stt, llm, tts))
    if not ok:
        st.error(
            f"vLLM não respondeu em {cfg.llm.server_url}. Suba o servidor primeiro "
            "(`.venv/bin/python deployment/serve_vllm.py --exec`) e recarregue a página."
        )
        st.stop()
    st.session_state.ready = True

for turn in st.session_state.turns:
    with st.chat_message(turn["role"]):
        st.write(turn["text"])
        if turn.get("image") is not None:
            st.image(turn["image"], width=180)
        if turn.get("audio"):
            st.audio(turn["audio"])
        if turn.get("latency"):
            st.caption(turn["latency"])

if st.session_state.turns and st.button("🗑️ Limpar conversa"):
    st.session_state.session = Session(system_prompt=SYSTEM_PROMPT, cfg=cfg.llm)
    st.session_state.turns = []
    st.rerun()

st.divider()

mode = st.radio("Entrada", ["🎤 Voz (grava e transcreve)", "⌨️ Texto (pula o STT)"], horizontal=True)
image_value = st.camera_input("Opcional: mostre algo à câmera para este turno (docs/05 §3)")

user_text: str | None = None
frames: list[VideoFrame] = []
pil_image = None
if image_value is not None:
    pil_image = Image.open(io.BytesIO(image_value.getvalue())).convert("RGB")
    frames = [VideoFrame(image=pil_image, frame_id=0, t=time.time())]

if mode.startswith("🎤"):
    audio_value = st.audio_input("Grave sua pergunta")
    if audio_value is not None and st.button("Enviar", type="primary", use_container_width=True):
        t0 = time.perf_counter()
        pcm16k = _decode_audio_to_16k(audio_value.getvalue())
        with st.spinner("Transcrevendo (Parakeet-TDT)..."):
            transcript = asyncio.run(_transcribe(stt, pcm16k))
        user_text = transcript.text
        st.session_state["_t0"] = t0
        st.session_state["_t_stt"] = time.perf_counter()
else:
    typed = st.text_input("Digite sua pergunta")
    if typed and st.button("Enviar", type="primary", use_container_width=True):
        user_text = typed
        st.session_state["_t0"] = time.perf_counter()
        st.session_state["_t_stt"] = st.session_state["_t0"]

if user_text:
    st.session_state.turns.append({"role": "user", "text": user_text, "image": pil_image})

    with st.spinner("Pensando (LFM2.5-VL-3B)..."):
        response_text = asyncio.run(_respond(st.session_state.session, llm, user_text, frames, registry))
    t_llm = time.perf_counter()

    with st.spinner("Sintetizando voz (XTTS-v2)..."):
        pcm_bytes = asyncio.run(_speak(tts, response_text or "Desculpe, não consegui responder."))
    t_tts = time.perf_counter()

    wav_bytes = _pcm_f32_to_wav_bytes(np.frombuffer(pcm_bytes, dtype=np.float32), TTS_SAMPLE_RATE)

    t0, t_stt = st.session_state["_t0"], st.session_state["_t_stt"]
    latency = (
        f"STT {(t_stt - t0) * 1000:.0f}ms · LLM {(t_llm - t_stt) * 1000:.0f}ms · "
        f"TTS {(t_tts - t_llm) * 1000:.0f}ms · total {(t_tts - t0) * 1000:.0f}ms"
    )
    st.session_state.turns.append({"role": "assistant", "text": response_text, "audio": wav_bytes, "latency": latency})
    st.rerun()
