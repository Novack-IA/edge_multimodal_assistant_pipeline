"""Contrato de turno/sessão — o estado explícito que docs/09 exige.

O KV cache de token é do vLLM; isto aqui é o objeto com ciclo de vida
documentado que decide o que entra na próxima requisição e o que é
consolidado no histórico persistido (CLAUDE.md invariante 5).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from config import LlmConfig
from contracts.messages import ConsolidatedTurn, OpenAIMessage


@dataclass
class Session:
    system_prompt: str
    cfg: LlmConfig
    history: list[ConsolidatedTurn] = field(default_factory=list)

    @property
    def system_and_history(self) -> list[OpenAIMessage]:
        """Prefixo estável `[SYSTEM][HISTÓRICO]` — texto puro, sem imagens
        (docs/09 §3). Idêntico byte-a-byte entre turnos consecutivos até o
        próximo commit, maximizando o Automatic Prefix Caching do vLLM."""
        messages: list[OpenAIMessage] = [{"role": "system", "content": self.system_prompt}]
        for turn in self._windowed_history():
            messages.append({"role": "user", "content": turn.user_text})
            messages.append({"role": "assistant", "content": turn.assistant_text})
        return messages

    def _windowed_history(self) -> list[ConsolidatedTurn]:
        if self.cfg.history_max_turns <= 0:
            return []
        return self.history[-self.cfg.history_max_turns :]

    def commit(self, user_text: str, assistant_text: str) -> None:
        """Único ponto de escrita no histórico — só alcançado no caminho de
        sucesso do turno (docs/09 §5/§9 regra 2). NUNCA chamar em barge-in."""
        self.history.append(
            ConsolidatedTurn(user_text=user_text, assistant_text=assistant_text, t=time.time())
        )

    def estimate_tokens(
        self,
        extra_text: str = "",
        n_image_frames: int = 0,
        video_token_budget: int = 0,
    ) -> int:
        """Heurística grosseira (~4 chars/token) para orçar ANTES de enviar
        (NFR-R7, docs/09 §6). Calibrar com medições reais de tokens-por-imagem
        do processor do vLLM em M1/M2 (docs/05 §5) — não é o valor final."""
        text_chars = (
            len(self.system_prompt)
            + sum(len(t.user_text) + len(t.assistant_text) for t in self._windowed_history())
            + len(extra_text)
        )
        return text_chars // 4 + n_image_frames * video_token_budget

    def fits_context(
        self,
        extra_text: str,
        n_image_frames: int,
        video_token_budget: int,
        reserve_for_response: int,
    ) -> bool:
        """Checagem preventiva do teto de 32K — nunca descobrir o estouro
        pela resposta de erro do vLLM (docs/09 §6)."""
        used = self.estimate_tokens(extra_text, n_image_frames, video_token_budget)
        return used + reserve_for_response <= self.cfg.context_length
