"""Ferramentas de referência — locais, sem I/O de rede (invariante 2 do
CLAUDE.md: chamadas externas só dentro de tools declaradas explicitamente,
assíncronas, não bloqueantes). Registrar ferramentas reais do produto aqui
ou em módulos irmãos e chamar `register_builtin_tools` no boot.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from contracts.messages import ToolSpec
from tools.registry import ToolRegistry

GET_CURRENT_TIME = ToolSpec(
    name="get_current_time",
    description="Retorna a data e hora atuais no fuso horário local do dispositivo.",
    parameters={"type": "object", "properties": {}, "required": []},
    timeout_ms=1000,
)


def register_builtin_tools(registry: ToolRegistry) -> None:
    @registry.register(GET_CURRENT_TIME)
    async def _get_current_time() -> str:
        await asyncio.sleep(0)  # cede o loop — ver invariante "nunca bloqueia" mesmo trivial
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
