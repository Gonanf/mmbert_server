"""Endpoints: contrato /v1/chat/completions (sin romper), /v1/classify nuevo, /health."""

from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient

import server

V_WAIT = np.array([1.0, 0, 0, 0], dtype=np.float32)      # stage1 -> wait
V_EXEC = np.array([-1.0, 0, 0, 0], dtype=np.float32)     # done + requiere


def _app_with(classifier, backend="gguf", model="fake:harrier"):
    server.app.state.classifier = classifier
    server.app.state.backend = backend
    server.app.state.model = model
    server._LATENCIES.clear()
    return TestClient(server.app)


def test_chat_completions_shape_preservado(heads_dir, fake_embedder):
    fake_embedder._vec = V_EXEC
    client = _app_with(server.HeadClassifier(fake_embedder, heads_dir))
    r = client.post("/v1/chat/completions", json={
        "model": "classifier",
        "messages": [{"role": "user", "content": "che, ¿me pasás el resumen?"}],
        "temperature": 0.0,
    })
    assert r.status_code == 200
    content = json.loads(r.json()["choices"][0]["message"]["content"])
    assert set(content) == {"category", "confidence"}
    assert content["category"] == "EXECUTE"
    assert isinstance(content["confidence"], float)


def test_chat_completions_puede_devolver_wait(heads_dir, fake_embedder):
    fake_embedder._vec = V_WAIT
    client = _app_with(server.HeadClassifier(fake_embedder, heads_dir))
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "y después me dijo que"}],
    })
    content = json.loads(r.json()["choices"][0]["message"]["content"])
    assert content["category"] == "WAIT"


def test_classify_shape_completa(heads_dir, fake_embedder):
    fake_embedder._vec = V_EXEC
    client = _app_with(server.HeadClassifier(fake_embedder, heads_dir))
    r = client.post("/v1/classify", json={
        "text": "che, ¿me pasás el resumen del deploy?",
        "context": ["hola", "¿cómo va?"],
        "agents": ["Jane", "Doktor"],
    })
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"stage1", "stage2", "verdict", "category", "confidence"}
    assert set(body["stage1"]) == {"label", "proba"}
    assert body["verdict"] == "exec"
    assert body["category"] == "EXECUTE"


def test_classify_valida_context_max_10(heads_dir, fake_embedder):
    client = _app_with(server.HeadClassifier(fake_embedder, heads_dir))
    r = client.post("/v1/classify", json={"text": "x", "context": ["a"] * 11})
    assert r.status_code == 400
    r2 = client.post("/v1/classify", json={"text": "x", "context": "no-lista"})
    assert r2.status_code == 400
    r3 = client.post("/v1/classify", json={"text": ""})
    assert r3.status_code == 400


def test_classify_requiere_cabezas(heads_dir, fake_embedder):
    client = _app_with(server.PrototypeClassifier(fake_embedder))
    r = client.post("/v1/classify", json={"text": "x"})
    assert r.status_code == 409
    assert "legacy" in r.json()["error"]


def test_health_reporta_stack_y_latencia(heads_dir, fake_embedder):
    fake_embedder._vec = V_EXEC
    client = _app_with(server.HeadClassifier(fake_embedder, heads_dir), backend="gguf", model="/gguf/harrier.Q8_0.gguf")
    for _ in range(3):
        client.post("/v1/classify", json={"text": "x"})
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert h["backend"] == "gguf"
    assert h["model"] == "/gguf/harrier.Q8_0.gguf"
    assert h["heads_loaded"] is True
    assert h["mode"] == "two-stage-heads"
    assert h["n_calls"] == 3
    assert isinstance(h["latency_p50_ms"], (int, float))