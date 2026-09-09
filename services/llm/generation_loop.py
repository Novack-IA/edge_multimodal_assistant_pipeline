"""Laço de geração normativo — docs/09 §8, quase literal.

Frames entram uma vez na mensagem do turno; barge-in não commita nada;
tool calls ficam no staging local do turno até um resultado final de texto
existir. Ver as 5 regras invioláveis em docs/09 §9.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from contracts.messages import (
    OpenAIMessage,
    SamplingCfg,
    ToolCall,
    ToolSpec,
    VideoFrame,
    frame_to_image_part,
)
from services.llm.client import LlmClient
from services.llm.session import Session
from tools.registry import ToolRegistry


async def run_turn(
    session: Session,
    llm_client: LlmClient,
    user_text: str,
    frames: list[VideoFrame],
    tools: list[ToolSpec],
    tool_registry: ToolRegistry,
    cancel_token: asyncio.Event,
) -> AsyncIterator[str]:
    turn_messages: list[OpenAIMessage] = [
        *session.system_and_history,  # prefixo estável — texto puro
        {
            "role": "user",
            "content": [
                *[frame_to_image_part(f) for f in frames],
                {"type": "text", "text": user_text},
            ],
        },
    ]
    sampling = SamplingCfg(
        temperature=session.cfg.temperature,
        top_p=session.cfg.top_p,
        tool_temperature=session.cfg.tool_temperature,
        max_new_tokens=session.cfg.max_new_tokens,
    )

    response_text = ""
    while True:
        made_tool_call = False
        async for event in llm_client.stream_chat(turn_messages, tools, sampling, cancel_token):
            if cancel_token.is_set():
                return  # BARGE-IN — nada commitado; sessão intacta (docs/09 §5/§9)

            if isinstance(event, ToolCall):
                result = await tool_registry.run(event, cancel_token)
                if cancel_token.is_set():
                    return
                turn_messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": event.id,
                                "type": "function",
                                "function": {
                                    "name": event.name,
                                    "arguments": json.dumps(event.arguments, ensure_ascii=False),
                                },
                            }
                        ],
                    }
                )
                turn_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "name": result.name,
                        "content": result.content,
                    }
                )
                made_tool_call = True
                continue  # nova chamada com o resultado anexado

            if event.is_tool_call:
                continue  # chunk parcial de tool call ainda acumulando no cliente

            if event.text:
                response_text += event.text
                yield event.text  # stream p/ TTS (docs/07)

        if not made_tool_call:
            break

    session.commit(user_text, response_text)  # SÓ texto entra no histórico (docs/09 §9 regra 3)
