"""Contratos entre estágios do laço (ver CLAUDE.md e docs/01, docs/09).

Um estágio nunca depende do formato interno de outro — todo dado que cruza
uma fronteira de estágio é uma das dataclasses/TypedDicts deste pacote.
"""

from contracts.messages import (
    AudioChunk,
    AudioOut,
    LlmToken,
    OpenAIMessage,
    SamplingCfg,
    ToolCall,
    ToolResult,
    ToolSpec,
    Transcript,
    TurnState,
    VadEvent,
    VideoFrame,
)

__all__ = [
    "AudioChunk",
    "AudioOut",
    "LlmToken",
    "OpenAIMessage",
    "SamplingCfg",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "Transcript",
    "TurnState",
    "VadEvent",
    "VideoFrame",
]
