"""Captura + amostragem de frames — SEM encoder próprio (docs/05, docs/09).

A codificação de visão acontece dentro do forward do LFM2.5-VL-3B, servido
pelo vLLM — nosso código só produz e enfileira imagens brutas (PIL).
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Literal

import numpy as np
from PIL import Image

from config import VisionConfig
from contracts.messages import VideoFrame


class FrameRingBuffer:
    """Buffer compartilhado de frames brutos, atualizado continuamente pelo
    produtor de vídeo, independente do estado da conversa (docs/08 §6)."""

    def __init__(self, maxlen: int) -> None:
        self._buf: deque[VideoFrame] = deque(maxlen=maxlen)

    def push(self, frame: VideoFrame) -> None:
        self._buf.append(frame)

    def snapshot(
        self,
        n: int,
        selection: Literal["uniform", "keyframe"] = "uniform",
    ) -> list[VideoFrame]:
        """Tirado na entrada em THINKING; congelado durante a geração do turno
        (docs/09 §4). `n` = `vision.window_frames`."""
        frames = list(self._buf)
        if not frames:
            return []
        if selection == "keyframe":
            # Seleção por diferença de frame — fica para M5 (docs/05 §5, docs/10).
            raise NotImplementedError("frame_selection=keyframe é M5 — ver docs/10-roadmap-milestones.md")
        if len(frames) <= n:
            return frames
        idx = sorted(set(np.linspace(0, len(frames) - 1, n).round().astype(int).tolist()))
        return [frames[i] for i in idx]

    def __len__(self) -> int:
        return len(self._buf)


class CameraProducer:
    """Produtor contínuo que amostra a câmera no `encode_fps` configurado e
    empilha frames brutos (redimensionados) no ring buffer.

    TODO (docs/03 §5): trocar `cv2.VideoCapture(0)` por um pipeline
    GStreamer com NVMM para captura CSI acelerada sem cópia CPU↔GPU — hoje
    isto é um caminho USB/genérico suficiente para desenvolvimento.
    """

    def __init__(self, cfg: VisionConfig, ring: FrameRingBuffer) -> None:
        self._cfg = cfg
        self._ring = ring
        self._frame_id = 0

    async def run(self, stop: asyncio.Event) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "opencv-python não instalado — rode `pip install -e .[vision]` no device "
                "antes de capturar vídeo de verdade (docs/03 §5)."
            ) from exc

        cap = cv2.VideoCapture(0)
        try:
            interval_s = 1.0 / self._cfg.encode_fps
            while not stop.is_set():
                ok, bgr = cap.read()
                if ok:
                    self._ring.push(self._to_frame(bgr, cv2))
                await asyncio.sleep(interval_s)
        finally:
            cap.release()

    def _to_frame(self, bgr: np.ndarray, cv2) -> VideoFrame:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        # Alinhado ao tiling nativo do modelo (512x512 — docs/05 §2).
        image.thumbnail((self._cfg.image_max_side_px, self._cfg.image_max_side_px))
        frame = VideoFrame(image=image, frame_id=self._frame_id, t=time.time())
        self._frame_id += 1
        return frame
