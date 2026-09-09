"""Dataclasses/TypedDicts trocadas entre estágios do laço.

Espelha exatamente o esboço de docs/01-architecture.md §6, estendido com os
tipos auxiliares que docs/06 (tool calling), docs/07 (TTS) e docs/09
(contrato de turno/sessão) exigem. Mudar um destes tipos é uma decisão de
arquitetura — ver CLAUDE.md invariante 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypedDict

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# Áudio / VAD / STT
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """microfone → VAD/STT"""

    pcm: bytes
    sample_rate: int
    t_capture: float


@dataclass(frozen=True, slots=True)
class VadEvent:
    """VAD → orquestrador"""

    kind: Literal["speech_start", "speech_end"]
    t: float


@dataclass(frozen=True, slots=True)
class Transcript:
    """STT → orquestrador"""

    text: str
    is_final: bool
    t_start: float
    t_end: float


# --------------------------------------------------------------------------
# Vídeo
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """captura → janela de vídeo (SEM encoder nosso — doc 05)"""

    image: Image.Image | np.ndarray
    frame_id: int
    t: float


# --------------------------------------------------------------------------
# LLM / vLLM — streaming e tool calling (doc 06)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LlmToken:
    """vLLM → orquestrador (stream SSE)"""

    text: str
    is_tool_call: bool
    done: bool


@dataclass(frozen=True, slots=True)
class ToolCall:
    """vLLM (via parser lfm2) → ferramentas"""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """ferramentas → vLLM (mensagem role="tool")"""

    tool_call_id: str
    name: str
    content: str
    is_error: bool
    t: float


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Definição de uma ferramenta exposta ao tool calling (registro em tools/)."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema dos argumentos
    timeout_ms: int

    def as_openai_tool(self) -> dict[str, Any]:
        """Formato esperado pelo campo `tools=` da API OpenAI-compatible do vLLM."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True, slots=True)
class SamplingCfg:
    """Parâmetros de amostragem por chamada ao vLLM (doc 06 §5)."""

    temperature: float = 0.5
    top_p: float = 0.9
    tool_temperature: float = 0.2
    max_new_tokens: int = 256


# --------------------------------------------------------------------------
# TTS (doc 07)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AudioOut:
    """TTS → playback (stream, interrompível)"""

    pcm: bytes
    is_last: bool


# --------------------------------------------------------------------------
# Mensagens no formato OpenAI-style enviadas ao vLLM (doc 05 §3, doc 09 §3)
# --------------------------------------------------------------------------


class ImageUrlPart(TypedDict):
    type: Literal["image_url"]
    image_url: dict[str, str]  # {"url": "data:image/...;base64,..."}


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class OpenAIMessage(TypedDict, total=False):
    """Uma mensagem da lista `messages` enviada a /v1/chat/completions.

    `content` é `str` para mensagens de texto puro (system, histórico, tool)
    ou `list[ImageUrlPart | TextPart]` para a mensagem do turno atual, que
    carrega os frames da janela de vídeo (doc 05 §3).
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ImageUrlPart | TextPart]
    tool_calls: list[dict[str, Any]]
    tool_call_id: str
    name: str


def frame_to_data_url(frame: VideoFrame, *, format: str = "JPEG") -> str:
    """Converte um VideoFrame em data URL base64 para content de imagem (doc 05 §3)."""
    import base64
    from io import BytesIO

    image = frame.image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buf = BytesIO()
    image.save(buf, format=format)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if format.upper() == "JPEG" else f"image/{format.lower()}"
    return f"data:{mime};base64,{b64}"


def frame_to_image_part(frame: VideoFrame) -> ImageUrlPart:
    return {"type": "image_url", "image_url": {"url": frame_to_data_url(frame)}}


# --------------------------------------------------------------------------
# Estado do turno (doc 08 §7)
# --------------------------------------------------------------------------


class TurnState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


@dataclass(slots=True)
class ConsolidatedTurn:
    """Um turno já commitado no histórico persistido — só texto (doc 09 §5)."""

    user_text: str
    assistant_text: str
    t: float = field(default=0.0)
