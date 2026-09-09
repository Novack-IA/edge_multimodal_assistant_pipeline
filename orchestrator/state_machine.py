"""Máquina de estados de turno + barge-in (docs/08 §2/§7).

IDLE → LISTENING → THINKING → SPEAKING → IDLE, com a transição especial
BARGE-IN a partir de THINKING/SPEAKING de volta a LISTENING. Independente
de quanto do Pipecat é reusado na integração real, este é o contrato de
comportamento (docs/08 §1).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from contracts.messages import TurnState, VadEvent

OnBargeIn = Callable[[], Awaitable[None]]


@dataclass
class Turn:
    state: TurnState = TurnState.IDLE
    cancel_token: asyncio.Event = field(default_factory=asyncio.Event)


class ConversationStateMachine:
    def __init__(self, on_barge_in: OnBargeIn | None = None) -> None:
        self.turn = Turn()
        self._on_barge_in = on_barge_in

    @property
    def state(self) -> TurnState:
        return self.turn.state

    def enter(self, state: TurnState) -> None:
        self.turn.state = state

    async def on_vad_event(self, ev: VadEvent) -> None:
        if ev.kind == "speech_start" and self.turn.state in (TurnState.THINKING, TurnState.SPEAKING):
            await self.barge_in()
        elif ev.kind == "speech_start" and self.turn.state == TurnState.IDLE:
            self.enter(TurnState.LISTENING)
        elif ev.kind == "speech_end" and self.turn.state == TurnState.LISTENING:
            self.enter(TurnState.THINKING)

    async def barge_in(self) -> None:
        """docs/08 §2:
        1. Seta cancel_token -> aborta a requisição HTTP do vLLM e a síntese do TTS
           (quem está lendo o token em `run_turn`/`synthesize_stream` observa e retorna).
        2. `_on_barge_in` esvazia a fila de playback / cancela tools em voo.
        3. Não commita nada — a sessão permanece no último estado consolidado
           (garantido por construção: `Session.commit` só roda no fim do turno).
        4. Volta para LISTENING.
        """
        self.turn.cancel_token.set()
        if self._on_barge_in is not None:
            await self._on_barge_in()
        self.enter(TurnState.LISTENING)
        # Novo cancel_token para o próximo turno — o antigo fica setado para
        # sempre como sinal para qualquer coroutine ainda em voo do turno abortado.
        self.turn.cancel_token = asyncio.Event()
