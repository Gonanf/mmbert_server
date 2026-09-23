"""Helpers compartidos: embedder falso determinista + cabezas sintéticas en tmp_path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import server


class FakeEmbedder:
    """Devuelve el mismo vector para todo texto (determinista, sin modelo)."""

    def __init__(self, vec: np.ndarray | None = None, dim: int = 4) -> None:
        self._vec = np.ones(dim, dtype=np.float32) if vec is None else np.asarray(vec, dtype=np.float32)
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(tuple(texts))
        return np.tile(self._vec, (len(texts), 1))

    def dim(self) -> int:
        return int(self._vec.shape[0])


def write_heads(directory: Path, *, dim: int = 4,
                coef1: np.ndarray | None = None, intercept1: float = 0.0, thr1: float = 0.5,
                coef2: np.ndarray | None = None, intercept2: float = 0.0, thr2: float = -0.5,
                cal_w: float = 2.0, cal_b: float = 0.0) -> Path:
    """Escribe stage1.npz/stage2.npz sintéticos controlables."""

    def c(coef) -> np.ndarray:
        return np.ones(dim, dtype=np.float32) if coef is None else np.asarray(coef, dtype=np.float32)

    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "stage1.npz",
             coef=c(coef1), intercept=np.float64(intercept1), thr=np.float64(thr1),
             cal_w=np.float64(cal_w), cal_b=np.float64(cal_b), dim=np.int32(dim),
             nombre="stage1_wait_asr_incompleto")
    np.savez(directory / "stage2.npz",
             coef=c(coef2), intercept=np.float64(intercept2), thr=np.float64(thr2),
             cal_w=np.float64(cal_w), cal_b=np.float64(cal_b), dim=np.int32(dim),
             nombre="stage2_no_requiere")
    return directory


@pytest.fixture()
def heads_dir(tmp_path: Path) -> Path:
    # coef1=[1,0,0,0]:  v=[1,0,0,0] -> s1=1 >= 0.5 -> wait
    #                  v=[0,...]   -> s1=0 <  0.5 -> done; s2=0 >= -0.5 -> no_requiere
    # coef2=[1,0,0,0]: v=[-1,0,0,0] -> s1=-1 -> done; s2=-1 < -0.5 -> requiere -> exec
    return write_heads(tmp_path)


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def classifier(heads_dir: Path, fake_embedder: FakeEmbedder) -> server.HeadClassifier:
    return server.HeadClassifier(fake_embedder, heads_dir)


@pytest.fixture()
def tmp_assets(tmp_path: Path) -> tuple[Path, FakeEmbedder]:
    return write_heads(tmp_path), FakeEmbedder()