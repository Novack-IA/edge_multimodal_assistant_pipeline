"""Wiring do laço completo: VAD → STT → vLLM (LFM2.5-VL-3B) → TTS, com
barge-in e janela de vídeo (docs/01 §3-4, docs/08, docs/09).

Este módulo é o ponto de integração real (M3/M4). A captura de áudio/mic e
o sink de playback ficam atrás de `AudioSource`/`AudioSink` — plugáveis com
Pipecat (docs/08 §1) ou um loop nativo simples; nenhum dos dois é
instanciado aqui para manter este módulo testável sem hardware.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import numpy as np

from config import AppConfig
from contracts.messages import AudioChunk, AudioOut, TurnState
from orchestrator.state_machine import ConversationStateMachine
from services.llm.client import LlmClient
from services.llm.generation_loop import run_turn
from services.llm.session import Session
from services.stt import create_stt_engine
from services.tts.worker import TtsEngine, XttsV2Engine
from services.vad.silero import SileroEndpointDetector
from services.vision.capture import CameraProducer, FrameRingBuffer
from tools.builtin import register_builtin_tools
from tools.registry import ToolRegistry

# Fronteiras de flush pro TTS (docs/07 §5 "segmentar por frase/cláusula").
# Medido no device (benchmarks/bench_roundtrip.py): esperar só por ".!?\n"
# faz o "1o áudio" depender do tamanho da frase inteira que o LLM decidir
# gerar (~1.6-1.8s medido para uma frase natural de ~60 tokens) — mesmo com
# TTFT de ~120ms. Literatura/prática de produção (pipelines de voz da
# LiveKit/Pipecat) corta também em vírgula/ponto-e-vírgula como fallback
# depois de um mínimo de caracteres, pra limitar o pior caso sem picotar
# frases curtas em fragmentos desnecessários.
SENTENCE_END = (".", "!", "?", "\n")
CLAUSE_END = (",", ";", ":")
MIN_CHARS_FOR_CLAUSE_FLUSH = 40


def is_speakable_chunk_boundary(buffer: str) -> bool:
    if not buffer:
        return False
    last = buffer[-1]
    if last in SENTENCE_END:
        return True
    return last in CLAUSE_END and len(buffer) >= MIN_CHARS_FOR_CLAUSE_FLUSH


def pcm_to_f32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


class Orchestrator:
    def __init__(self, cfg: AppConfig, system_prompt: str, *, tts_engine: TtsEngine | None = None) -> None:
        self.cfg = cfg
        self.session = Session(system_prompt=system_prompt, cfg=cfg.llm)
        self.llm_client = LlmClient(cfg.llm)
        self.tool_registry = ToolRegistry(cfg.tools)
        register_builtin_tools(self.tool_registry)
        self.vad_endpoint = SileroEndpointDetector(cfg.vad, sample_rate=cfg.audio.sample_rate)
        self.vad_bargein = SileroEndpointDetector(cfg.vad, sample_rate=cfg.audio.sample_rate)
        self.stt = create_stt_engine(cfg.stt)
        self.tts: TtsEngine = tts_engine if tts_engine is not None else XttsV2Engine(cfg.tts)
        self.ring = FrameRingBuffer(maxlen=max(cfg.vision.window_frames * 4, 32))
        self.camera = CameraProducer(cfg.vision, self.ring)
        self.sm = ConversationStateMachine(on_barge_in=self._on_barge_in)
        self.playback_queue: asyncio.Queue[AudioOut] = asyncio.Queue()
        self._ready = asyncio.Event()
        self._turn_audio: list[bytes] = []

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    async def _on_barge_in(self) -> None:
        # Esvazia a fila de playback já enfileirada — corta o áudio no meio (docs/08 §2).
        while not self.playback_queue.empty():
            self.playback_queue.get_nowait()
        self._turn_audio.clear()

    # ------------------------------------------------------------------
    # Boot / warm-up (NFR-R6, docs/13 §4)
    # ------------------------------------------------------------------

    async def boot(self, *, poll_s: float = 2.0) -> None:
        """Não aceitar turnos antes do vLLM reportar saúde + responder a um
        turno dummy, e antes do TTS aquecer. Falhar aqui é melhor que aceitar
        turnos silenciosamente antes do warm-up (NFR-R6)."""
        deadline = time.monotonic() + self.cfg.cold_start.vllm_warmup_budget_s
        while time.monotonic() < deadline:
            if await self.llm_client.health_check():
                break
            await asyncio.sleep(poll_s)
        else:
            raise TimeoutError("vLLM não respondeu /models dentro do orçamento de cold-start (docs/13 §4)")
        await self.llm_client.warmup()
        await self.tts.warmup()
        self._ready.set()

    # ------------------------------------------------------------------
    # Laço de áudio (LISTENING) — docs/04 §3, docs/08 §2-3
    # ------------------------------------------------------------------

    async def handle_audio_chunk(self, chunk: AudioChunk) -> None:
        if not self._ready.is_set():
            return
        detector = self.vad_bargein if self.sm.state == TurnState.SPEAKING else self.vad_endpoint
        ev = detector.process_chunk(chunk)

        if self.sm.state == TurnState.LISTENING:
            self._turn_audio.append(chunk.pcm)

        if ev is not None:
            prev_state = self.sm.state
            await self.sm.on_vad_event(ev)
            if prev_state == TurnState.LISTENING and self.sm.state == TurnState.THINKING:
                await self._run_turn_from_endpoint()
                self._turn_audio.clear()
            elif ev.kind == "speech_start" and self.sm.state == TurnState.LISTENING and prev_state != TurnState.LISTENING:
                self._turn_audio.clear()  # novo turno começando (inclui pós barge-in)

    # ------------------------------------------------------------------
    # THINKING → SPEAKING (docs/09 §8, docs/07 §5)
    # ------------------------------------------------------------------

    async def _run_turn_from_endpoint(self) -> None:
        pcm = b"".join(self._turn_audio)
        transcript = await self.stt.transcribe(pcm_to_f32(pcm), partial=False)
        if not transcript.text.strip():
            self.sm.enter(TurnState.IDLE)
            return

        frames = self.ring.snapshot(self.cfg.vision.window_frames, self.cfg.vision.frame_selection)
        cancel_token = self.sm.turn.cancel_token

        if not self.session.fits_context(
            transcript.text, len(frames), self.cfg.vision.video_token_budget, self.cfg.llm.max_new_tokens
        ):
            # Orçamento estourado antes de enviar (NFR-R7, docs/09 §6) — degradação
            # simples para v0: derruba metade do histórico mais antigo e tenta de novo.
            self.session.history = self.session.history[len(self.session.history) // 2 :]

        buffer = ""
        first_chunk = True
        async for token_text in run_turn(
            self.session,
            self.llm_client,
            transcript.text,
            frames,
            self.tool_registry.specs(),
            self.tool_registry,
            cancel_token,
        ):
            if cancel_token.is_set():
                return
            buffer += token_text
            if first_chunk and buffer.strip():
                self.sm.enter(TurnState.SPEAKING)
                first_chunk = False
            if is_speakable_chunk_boundary(buffer):
                await self._speak(buffer, cancel_token)
                buffer = ""

        if cancel_token.is_set():
            return
        if buffer.strip():
            await self._speak(buffer, cancel_token)
        if not cancel_token.is_set() and self.sm.state == TurnState.SPEAKING:
            self.sm.enter(TurnState.IDLE)

    async def _speak(self, text: str, cancel_token: asyncio.Event) -> None:
        async for out in self.tts.synthesize_stream(text, cancel_token):
            if cancel_token.is_set():
                return
            await self.playback_queue.put(out)

    # ------------------------------------------------------------------
    # Vídeo — roda continuamente independente do estado (docs/08 §6)
    # ------------------------------------------------------------------

    async def run_camera(self, stop: asyncio.Event) -> None:
        await self.camera.run(stop)

    async def drain_playback(self) -> AsyncIterator[AudioOut]:
        while True:
            out = await self.playback_queue.get()
            yield out
            if out.is_last:
                return
