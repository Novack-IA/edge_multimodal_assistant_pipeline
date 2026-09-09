"""Config única e tipada do assistente (CLAUDE.md "Convenções de código").

Nenhuma constante mágica espalhada pelo código — tudo tunável mora em
`config.yaml` (copiado de `config.example.yaml`) e é validado aqui. Ver
docs/02 (requisitos), docs/09 §6 (orçamento de contexto) e docs/13 (vLLM).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent


class PlatformConfig(BaseModel):
    device: Literal["thor"] = "thor"
    power_mode: str = "max"
    jetpack: str = "7.x"
    cuda: str = "13.2"
    vllm_version: str = "TBD"


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 30
    aec: bool = True
    playback_buffer_ms: int = 30


class VadConfig(BaseModel):
    threshold: float = 0.5
    min_silence_ms: int = 600
    min_speech_ms: int = 150
    speech_pad_ms: int = 100
    barge_in_min_ms: int = 300


class SttConfig(BaseModel):
    # engine="parakeet" (default, validado no device — ver docs/04 §9): NeMo +
    # Parakeet-TDT-0.6B-v3, PyTorch/CUDA nativo, RTFx 57-90x medido no Thor.
    # engine="faster_whisper": CTranslate2 — o wheel pip para aarch64 não traz
    # CUDA (só CPU); mantido como fallback documentado (docs/04 §8).
    engine: Literal["parakeet", "faster_whisper"] = "parakeet"
    model: str = "large-v3"  # usado só por engine=faster_whisper
    compute_type: str = "int8"  # usado só por engine=faster_whisper
    device: str = "cuda"
    language: str = "pt"
    partial_beam_size: int = 1
    final_beam_size: int = 5
    parakeet_model: str = "nvidia/parakeet-tdt-0.6b-v3"  # usado só por engine=parakeet


class VisionConfig(BaseModel):
    camera: Literal["csi", "usb"] = "csi"
    encode_fps: float = 2
    window_frames: int = 6
    frame_selection: Literal["uniform", "keyframe"] = "uniform"
    image_max_side_px: int = 512
    video_token_budget: int = 1024


class LlmConfig(BaseModel):
    backbone: str = "LiquidAI/LFM2.5-VL-3B"
    runtime: Literal["vllm", "llama_cpp"] = "vllm"
    server_url: str = "http://127.0.0.1:8000/v1"
    tool_call_parser: str = "lfm2"
    precision: str = "bf16"
    context_length: int = 32768  # teto real do modelo — não aumentar (doc09 §6, doc13 §3)
    max_new_tokens: int = 256
    temperature: float = 0.5
    top_p: float = 0.9
    tool_temperature: float = 0.2
    history_max_turns: int = 8
    summarize_old_turns: bool = True
    warmup_required: bool = True
    gpu_memory_utilization: float = 0.5


class TtsConfig(BaseModel):
    engine: str = "xtts_v2"
    language: str = "pt"
    speaker_wav: str = "assets/reference_voice.wav"
    use_deepspeed: bool = True
    temperature: float = 0.65
    repetition_penalty: float = 2.0
    top_k: int = 50
    top_p: float = 0.8
    speed: float = 1.0


class LatencyBudgetsConfig(BaseModel):
    """Falha o CI se estourar (doc 02 / doc 11). Valores em ms."""

    barge_in_p50: int = 120
    barge_in_p95: int = 200
    roundtrip_p50: int = 800
    roundtrip_p95: int = 1200
    llm_ttft_p50: int = 250
    tts_ttfb_p50: int = 200
    vision_prefill_p50: int = 40


class ColdStartConfig(BaseModel):
    vllm_warmup_budget_s: int = 900


class ToolsConfig(BaseModel):
    timeout_ms: int = 4000


class LicensingConfig(BaseModel):
    org_annual_revenue_usd: str = "TBD"
    xtts_v2_cpml_reviewed: bool = False


class AppConfig(BaseModel):
    platform: PlatformConfig = Field(default_factory=PlatformConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    latency_budgets_ms: LatencyBudgetsConfig = Field(default_factory=LatencyBudgetsConfig)
    cold_start: ColdStartConfig = Field(default_factory=ColdStartConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    licensing: LicensingConfig = Field(default_factory=LicensingConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Carrega e valida `config.yaml`.

    Se `path` não for passado, procura `config.yaml` na raiz do repo e cai
    para `config.example.yaml` se ele ainda não existir (dev fresh-clone).
    """
    if path is not None:
        candidate = Path(path)
    else:
        local = REPO_ROOT / "config.yaml"
        candidate = local if local.exists() else REPO_ROOT / "config.example.yaml"

    with open(candidate, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)
