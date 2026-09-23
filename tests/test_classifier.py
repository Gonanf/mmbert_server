"""El clasificador de dos etapas: umbrales, veredicto, espejo y confianza."""

from __future__ import annotations

import numpy as np
import pytest

import server
from tests.conftest import FakeEmbedder

V_WAIT = np.array([1.0, 0, 0, 0], dtype=np.float32)      # stage1 -> wait
V_DONE_NO_REQ = np.array([0.0, 0, 0, 0], dtype=np.float32)  # done + no_requiere
V_EXEC = np.array([-1.0, 0, 0, 0], dtype=np.float32)     # done + requiere


def test_stage1_wait_domina(clf_factory):
    clf = clf_factory(V_WAIT)
    r = clf.classify("y después me dijo que")
    assert r["stage1"]["label"] == "wait"
    assert r["verdict"] == "wait"
    assert r["category"] == "WAIT"
    # stage2 se reporta igual (riesgo nulo, un dot product), pero no decide
    assert r["stage2"]["label"] in ("no_requiere", "requiere")
    assert 0.0 < r["confidence"] < 1.0
    assert 0.0 < r["stage1"]["proba"] < 1.0


def test_umbral1_por_debajo_es_done(clf_factory):
    clf = clf_factory(V_DONE_NO_REQ)
    r = clf.classify("dale, ya está")
    assert r["stage1"]["label"] == "done"
    assert r["verdict"] == "ignore"
    assert r["category"] == "IGNORE_SELF_TALK"


def test_verdict_exec_y_categoria(clf_factory):
    clf = clf_factory(V_EXEC)
    r = clf.classify("che, ¿me pasás el resumen del deploy?")
    assert r["stage1"]["label"] == "done"
    assert r["stage2"]["label"] == "requiere"
    assert r["verdict"] == "exec"
    assert r["category"] == "EXECUTE"
    assert r["confidence"] == r["stage2"]["proba"]


def test_espejo_third_party_por_agente(clf_factory):
    clf = clf_factory(V_DONE_NO_REQ)
    r = clf.classify("Jane dijo que ya lo ve", agents=["Jane", "Doktor"])
    assert r["category"] == "IGNORE_THIRD_PARTY"
    r2 = clf.classify("Jane dijo que ya lo ve", agents=["Doktor"])
    assert r2["category"] == "IGNORE_SELF_TALK"  # no menciona ningún agente conocido


def test_confianza_monotona_con_la_decision(clf_factory):
    # score = 2 -> proba mayor que score = 0.5 (misma calibración cal_w=2)
    hi = clf_factory(np.array([2.0, 0, 0, 0], dtype=np.float32))
    lo = clf_factory(np.array([0.5, 0, 0, 0], dtype=np.float32))
    assert hi.classify("x")["stage1"]["proba"] > lo.classify("x")["stage1"]["proba"]


def test_cabezas_faltantes_error_claro(tmp_path):
    emb = type("E", (), {"dim": lambda self: 4, "embed": lambda self, t: np.zeros((len(t), 4))})()
    with pytest.raises(RuntimeError, match="faltan cabezas"):
        server.HeadClassifier(emb, tmp_path)


def test_dims_incompatibles_error_claro(heads_dir, tmp_path):
    # cabezas 4-d vs embedder 6-d
    emb = type("E", (), {"dim": lambda self: 6, "embed": lambda self, t: np.zeros((len(t), 6))})()
    with pytest.raises(RuntimeError, match="dimensiones incompatibles"):
        server.HeadClassifier(emb, heads_dir)


# -- factory helper ----------------------------------------------------------

@pytest.fixture()
def clf_factory(heads_dir):
    def make(vec):
        return server.HeadClassifier(FakeEmbedder(vec), heads_dir)
    return make