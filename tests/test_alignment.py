"""Alineación embeddings/labels del banco: falla si los archivos no coinciden.

Sin GPU ni modelo: lee los .npy/.json ya calculados. Invariantes congeladas de la
medición (2497 muestras, 29 escenarios). Si el banco no está, se salta con aviso.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

TEXTOS = Path("/run/media/chaos/terciario/proyectos/kateto-train/out/eot-embeddings")
BANCO = Path("/run/media/chaos/terciario/proyectos/kateto-medicion/banco-eot/out/harrier-0.6b.npy")

NO_TERMINO = {"wait_asr_incompleto"}
NO_REQUIERE = {"no_response_al_aire", "no_response_otra_voz", "no_llenar_silencio"}
REQUIERE = {"solo_charla", "si_responde_indirecto", "silencio_charla_cerrada"}


def _banco_presente() -> bool:
    return BANCO.exists() and (TEXTOS / "textos.npy").exists() and (TEXTOS / "escenarios.json").exists()


def test_embeddings_y_labels_alineados():
    if not _banco_presente():
        pytest.skip("banco ausente — la alineación se verifica donde corre el entrenamiento")

    esc = json.load(open(TEXTOS / "escenarios.json"))
    textos = np.load(TEXTOS / "textos.npy", allow_pickle=True)
    X = np.load(BANCO)

    # Invariantes de la medición (produjo harrier-0.6b.npy y escenarios.json):
    # si alguna se rompe, algo se desalineó o se regeneró el banco sin avisar.
    assert len(textos) == len(esc) == X.shape[0] == 2497
    assert X.shape[1] == 1024, "harrier-0.6b es 1024-d"
    assert X.dtype == np.float32

    n_wait = sum(1 for e in esc if e in NO_TERMINO)
    idx2 = [i for i, e in enumerate(esc) if e in NO_REQUIERE or e in REQUIERE]
    n_no_req = sum(1 for i in idx2 if esc[i] in NO_REQUIERE)
    assert n_wait == 110, f"esperado 110 wait_asr_incompleto, hay {n_wait}"
    assert len(idx2) == 695, f"esperado 695 muestras de etapa 2, hay {len(idx2)}"
    assert n_no_req == 357, f"esperado 357 no_requiere, hay {n_no_req}"

    # El banco está normalizado (así se entrenó: normalizar() en audit-ridge.py)
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), "embeddings deben venir L2-normalizados"


def test_cabezas_guardadas_coinciden_con_el_banco():
    """Las cabezas entrenadas deben ser 1024-d o el server no arranca."""
    heads = Path(__file__).parent.parent / "heads"
    if not (heads / "stage1.npz").exists():
        pytest.skip("cabezas no entrenadas todavía")
    for name in ("stage1.npz", "stage2.npz"):
        z = np.load(heads / name)
        assert int(z["dim"]) == 1024, f"{name} espera {z['dim']}-d, el banco es 1024-d"
    assert (heads / "metrics.json").exists(), "falta metrics.json (entregable de train_heads.py)"