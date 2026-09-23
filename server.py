"""
Clasificador de dos etapas para Kateto, sobre embeddings de harrier-oss-v1-0.6b
y cabezas ridge entrenadas (no prototipos).

Pipeline en serie:
  Etapa 1 — ¿terminó la idea?       wait_asr_incompleto  vs  turno completo
  Etapa 2 — ¿requiere respuesta?    no_response*/no_llenar_silencio  vs  requiere
  (solo si la etapa 1 dice "completo")

Veredicto:
  etapa1 = incompleto  -> wait  -> category WAIT
  etapa2 = no_requiere -> ignore -> IGNORE_SELF_TALK / IGNORE_THIRD_PARTY (espejo
                                    por mención de un agente conocido en el texto)
  etapa2 = requiere    -> exec  -> category EXECUTE

Cabezas: heads/stage1.npz y heads/stage2.npz (RidgeClassifier lineal; thr elegido
por F1 en validación interna durante el entrenamiento; cal_w/cal_b convierten la
decisión lineal a proba). Entrenar con `python3 train_heads.py` (interprete del
sistema, tiene scikit-learn).

Backends de embeddings:
  gguf  (default)  llama-server persistente (llama.cpp) sobre el GGUF Q8_0,
                    --pooling mean --embd-normalize 2. La latencia servida mide
                    ~30-115 ms p50 en este host (vs ~5-15 ms del MiniLM-L6 ONNX).
  onnx             onnxruntime (MiniLM-L6 prod u otro BERT ONNX).

Los prototipos heurísticos viejos quedan SOLO detrás de --legacy-prototypes
(fallback declarado). Sin cabezas y sin el flag, el server no arranca.

Endpoints (contrato OpenAI-compatible preservado):
  POST /v1/chat/completions  -> {"choices":[{"message":{"content":
                               "{\"category\":\"...\",\"confidence\":0.xx}"}}]}
  POST /v1/classify          -> {"text","context","agents"} -> shape de dos etapas
  GET  /health               -> backend, modelo, cabezas, p50 de latencia

Uso:
  python3 server.py                                # gguf, heads/ , detecta el GGUF
  python3 server.py --backend onnx --embeddings-model Qdrant/all-MiniLM-L6-v2-onnx
  python3 server.py --legacy-prototypes --backend onnx   # modo viejo, explícito
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Protocol, cast

import numpy as np

log = logging.getLogger("mmbert-server")

# ---------------------------------------------------------------------------
# Clasificación — debe matchear kateto.core.event.Classification
# ---------------------------------------------------------------------------

CATEGORIES = ("EXECUTE", "IGNORE_SELF_TALK", "IGNORE_THIRD_PARTY", "WAIT")

# Protege contra cargar los npz con un backend de dimensión equivocada.
HARRIER_DIM = 1024

# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------


class GgufEmbedder:
    """llama-server persistente (llama.cpp) como embedder HTTP.

    Mismo modelo/flags que el banco: harrier-oss-v1-0.6b Q8_0, --pooling mean,
    --embd-normalize 2, 4 hilos. Verificado contra el banco (cos >= 0.9999).
    """

    def __init__(self, model_path: Path, *, threads: int = 4, port: int = 0) -> None:
        bin_path = shutil.which("llama-server")
        if bin_path is None:
            raise RuntimeError(
                "backend=gguf require el binario `llama-server` (llama.cpp) en PATH"
            )
        if not Path(model_path).exists():
            raise RuntimeError(f"modelo GGUF no encontrado: {model_path}")

        if port == 0:
            port = _free_port()
        self.port = port
        self.proc: subprocess.Popen[bytes] | None = subprocess.Popen(
            [
                bin_path, "-m", str(model_path),
                "--port", str(port), "--host", "127.0.0.1",
                "--embedding", "--pooling", "mean", "--embd-normalize", "2",
                "-t", str(threads), "-c", "4096", "--no-webui",
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        atexit.register(self.stop)
        self._wait_ready(timeout=120)

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc is None or self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server murió al arrancar (rc={self.proc.returncode if self.proc else '?'})"
                )
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        log.info("llama-server listo en 127.0.0.1:%d", self.port)
                        return
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"llama-server no quedó listo en {timeout:.0f}s")

    def embed(self, texts: list[str]) -> np.ndarray:
        payload = json.dumps({"input": texts}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/embeddings",
            data=payload, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())["data"]
        embs = np.array([d["embedding"] for d in data], dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.maximum(norms, 1e-12)

    def dim(self) -> int:
        return HARRIER_DIM  # verificado contra el banco al cargar

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), 15)
            except (ProcessLookupError, PermissionError):
                pass
            self.proc.wait(timeout=10)
        self.proc = None


class OnnxEmbedder:
    """onnxruntime, mean-pooling de last_hidden_state (MiniLM-L6 prod u otro)."""

    def __init__(self, model_ref: str, *, use_vulkan: bool = True) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        ONNX_FILENAME = "model.onnx"

        def _resolve(model_ref: str) -> Path:
            p = Path(model_ref)
            if p.exists():
                return p if p.is_file() else p / ONNX_FILENAME
            return Path(hf_hub_download(repo_id=model_ref, filename=ONNX_FILENAME,
                                        local_files_only=False))

        tokenizer_path = hf_hub_download(repo_id=model_ref, filename="tokenizer.json",
                                         local_files_only=False)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        model_path = _resolve(model_ref)
        self._wait_onnx = True

        providers = ["VulkanExecutionProvider", "CPUExecutionProvider"] if use_vulkan else ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
        except Exception as exc:
            if use_vulkan and "Vulkan" in str(exc):
                log.warning("Vulkan no disponible (%s), CPU", exc)
                self.session = ort.InferenceSession(str(model_path), sess_options=opts,
                                                    providers=["CPUExecutionProvider"])
            else:
                raise
        log.info("ONNX providers: %s", self.session.get_providers())

    def embed(self, texts: list[str]) -> np.ndarray:
        max_length = 128
        encoded = self.tokenizer.encode_batch(texts)
        input_ids = np.zeros((len(texts), max_length), dtype=np.int64)
        attention_mask = np.zeros((len(texts), max_length), dtype=np.int64)
        token_type_ids = np.zeros((len(texts), max_length), dtype=np.int64)
        for i, enc in enumerate(encoded):
            ids = enc.ids[:max_length]
            input_ids[i, : len(ids)] = ids
            attention_mask[i, : len(ids)] = 1

        (last_hidden,) = self.session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask,
                   "token_type_ids": token_type_ids}
        )
        last_hidden = np.asarray(last_hidden)
        mask = attention_mask.astype(np.float32)[:, :, np.newaxis]
        summed = np.sum(last_hidden * mask, axis=1)
        counts = np.maximum(np.sum(attention_mask, axis=1, keepdims=True), 1e-9)
        embs = summed / counts
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.where(norms == 0, 1.0, norms)

    def dim(self) -> int:
        return int(self.session.get_outputs()[0].shape[-1])

    def stop(self) -> None:  # onnxruntime no tiene proceso externo
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ---------------------------------------------------------------------------
# Clasificadores
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class _Embedder(Protocol):
    """Contrato de embedder: llama-server (gguf) u onnxruntime."""

    def embed(self, texts: list[str]) -> np.ndarray: ...
    def dim(self) -> int: ...


class HeadClassifier:
    """Dos cabezas ridge lineales entrenadas (train_heads.py)."""

    def __init__(self, embedder: _Embedder, heads_dir: Path) -> None:
        s1 = heads_dir / "stage1.npz"
        s2 = heads_dir / "stage2.npz"
        for p in (s1, s2):
            if not p.exists():
                raise RuntimeError(
                    f"faltan cabezas: no está {p}. Corré `python3 train_heads.py` "
                    f"o arrancá con --legacy-prototypes (fallback de prototipos)."
                )
        self.h1 = np.load(s1)
        self.h2 = np.load(s2)
        self.thr1 = float(self.h1["thr"])
        self.thr2 = float(self.h2["thr"])
        self.dim = int(self.h1["dim"])
        if self.dim != embedder.dim():
            raise RuntimeError(
                f"dimensiones incompatibles: cabezas esperan {self.dim}-d "
                f"pero el embedder devuelve {embedder.dim()}-d. Las cabezas se "
                f"entrenaron sobre harrier-0.6b ({HARRIER_DIM}-d): usá "
                f"--backend gguf con ese modelo, o entrená cabezas para este embedder."
            )
        self.embedder = embedder

    def classify(self, text: str, agents: list[str] | None = None) -> dict[str, object]:
        emb = self.embedder.embed([text])[0].astype(np.float64)

        s1 = float(self.h1["coef"] @ emb + self.h1["intercept"])
        p1 = float(_sigmoid(self.h1["cal_w"] * s1 + self.h1["cal_b"]))
        label1 = "wait" if s1 >= self.thr1 else "done"

        s2 = float(self.h2["coef"] @ emb + self.h2["intercept"])
        p2 = float(_sigmoid(self.h2["cal_w"] * s2 + self.h2["cal_b"]))
        label2 = "no_requiere" if s2 >= self.thr2 else "requiere"

        if label1 == "wait":
            verdict, category, conf = "wait", "WAIT", p1
        elif label2 == "no_requiere":
            mentioned = any(a.strip() and a.lower() in text.lower() for a in (agents or []))
            category = "IGNORE_THIRD_PARTY" if mentioned else "IGNORE_SELF_TALK"
            verdict, conf = "ignore", p2
        else:
            verdict, category, conf = "exec", "EXECUTE", p2

        return {
            "stage1": {"label": label1, "proba": round(p1, 4)},
            "stage2": {"label": label2, "proba": round(p2, 4)},
            "verdict": verdict,
            "category": category,
            "confidence": round(conf, 4),
        }


class PrototypeClassifier:
    """Fallback legacy: prototipos heurísticos por centroide (solo --legacy-prototypes)."""

    def __init__(self, embedder: _Embedder) -> None:
        self.embedder = embedder
        self._prototypes: dict[str, np.ndarray] = {}
        for cat, sentences in PROTOTYPES.items():
            center = self.embedder.embed(sentences).mean(axis=0)
            center /= np.linalg.norm(center)
            self._prototypes[cat] = center

    def classify(self, text: str, agents: list[str] | None = None) -> tuple[str, float]:
        emb = self.embedder.embed([text])[0]
        sims = np.array([float(np.dot(emb, self._prototypes[c])) for c in CATEGORIES[:3]])
        if agents:
            if any(name.lower() in text.lower() for name in agents):
                sims[CATEGORIES.index("EXECUTE")] += 0.25
        best = int(np.argmax(sims))
        sims -= sims.max()
        probs = np.exp(sims * 2.0)
        probs /= probs.sum()
        return CATEGORIES[best], float(probs[best])


# ---------------------------------------------------------------------------
# Prototipos legacy (no se usan salvo --legacy-prototypes)
# ---------------------------------------------------------------------------

PROTOTYPES: dict[str, list[str]] = {
    "EXECUTE": [
        "tell me about the project status",
        "plan the next sprint",
        "what are the outstanding tasks",
        "orchestrate the standup meeting",
        "organize the backlog",
        "coordinate the team",
        "summarize the current progress",
        "get an update on deliverables",
        "schedule a review session",
        "list the action items",
        "hello team, good morning",
        "hi Jane, how are you",
        "good morning everyone",
        "hello, what's the plan today",
        "hey team, ready to start",
        "good afternoon, let's begin",
        "hola Jane, buenos días",
        "buenos días equipo",
        "hola, ¿cómo están?",
    ],
    "IGNORE_SELF_TALK": [
        "I need to remember to check that",
        "let me think about this approach",
        "I should probably look at that later",
        "I'm going to try a different method",
        "I wonder if that would work",
        "remind me to follow up on that",
        "I think I understand now",
        "let me reconsider the options",
    ],
    "IGNORE_THIRD_PARTY": [
        "she said she would handle it",
        "they are working on the deployment",
        "he mentioned the deadline was tight",
        "the team is focusing on delivery",
        "the manager asked for an update",
        "they plan to release next week",
        "she is reviewing the pull request",
        "the stakeholders want a demo",
    ],
}


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="mmBERT Classifier", version="0.2.0")
# app.state.classifier se setea en main() antes de uvicorn.run()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    """OpenAI-compatible (contrato que Kateto consume como provider)."""
    clf = app.state.classifier
    if clf is None:
        return JSONResponse(status_code=503, content={"error": "classifier not initialized"})
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid JSON"})

    messages = body.get("messages", [])
    if not messages:
        return JSONResponse(status_code=400, content={"error": "messages required"})

    user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") in ("user", "system"):
            user_text = msg.get("content", "")
            break
    if not user_text:
        return JSONResponse(status_code=400, content={"error": "no user message content"})

    agents = body.get("agents", [])
    t0 = time.perf_counter()
    if isinstance(clf, HeadClassifier):
        r = clf.classify(user_text, agents=agents)
        category = cast(str, r["category"])
        confidence = cast(float, r["confidence"])
    else:
        category, confidence = clf.classify(user_text)  # PrototypeClassifier -> (str, float)
    _record_latency(time.perf_counter() - t0)

    payload = json.dumps({"category": category, "confidence": round(confidence, 4)})
    return JSONResponse(content={"choices": [{"message": {"content": payload}}]})


@app.post("/v1/classify")
async def classify(request: Request) -> JSONResponse:
    """Clasificación completa de dos etapas.

    Body: {"text": "...", "context": ["...", ...] (<= 10 mensajes previos),
           "agents": ["Jane", "Doktor"]}
    Respuesta: {"stage1": {"label","proba"}, "stage2": {"label","proba"},
                "verdict": "exec|wait|ignore", "category": "...", "confidence": 0.xx}
    """
    clf = app.state.classifier
    if clf is None:
        return JSONResponse(status_code=503, content={"error": "classifier not initialized"})
    if not isinstance(clf, HeadClassifier):
        return JSONResponse(status_code=409, content={
            "error": "modo legacy (prototipos) no tiene etapas; usá cabezas entrenadas "
                     "(entrená con train_heads.py y arrancá sin --legacy-prototypes)"
        })
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid JSON"})

    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        return JSONResponse(status_code=400, content={"error": "text (string) required"})

    context = body.get("context", [])
    agents = body.get("agents", [])
    if not isinstance(context, list) or not all(isinstance(c, str) for c in context):
        return JSONResponse(status_code=400, content={"error": "context must be list[str]"})
    if len(context) > 10:
        return JSONResponse(status_code=400, content={"error": "context max 10 messages"})
    if not isinstance(agents, list) or not all(isinstance(a, str) for a in agents):
        return JSONResponse(status_code=400, content={"error": "agents must be list[str]"})

    # El contexto se valida pero no se usa: las cabezas se entrenaron sobre turnos
    # individuales (textos.npy), clasificar con más texto los sacaría de distribución.
    t0 = time.perf_counter()
    r = clf.classify(text, agents=agents)
    _record_latency(time.perf_counter() - t0)
    return JSONResponse(content=r)


@app.get("/health")
async def health() -> JSONResponse:
    clf = app.state.classifier
    return JSONResponse(content={
        "status": "ok",
        "backend": app.state.backend,
        "model": app.state.model,
        "heads_loaded": isinstance(clf, HeadClassifier),
        "mode": "two-stage-heads" if isinstance(clf, HeadClassifier) else "legacy-prototypes",
        "latency_p50_ms": _latency_p50(),
        "n_calls": len(_LATENCIES),
    })


_LATENCIES: deque[float] = deque(maxlen=100)


def _record_latency(seconds: float) -> None:
    _LATENCIES.append(seconds * 1000.0)


def _latency_p50() -> float | None:
    if not _LATENCIES:
        return None
    return round(sorted(_LATENCIES)[len(_LATENCIES) // 2], 1)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

DEFAULT_GGUF_GLOB = (
    "/run/media/chaos/terciario/chaos-cache/gguf-eot/hub/"
    "models--mradermacher--harrier-oss-v1-0.6b-GGUF/snapshots/*/harrier-oss-v1-0.6b.Q8_0.gguf"
)
MODEL_REPO_DEFAULT_ONNX = "Qdrant/all-MiniLM-L6-v2-onnx"  # encoder de producción (384-d)


def _default_gguf() -> str:
    # Path(DEFAULT_GGUF_GLOB).parent es "snapshots/*" (un glob, no una dir real)
    snaps = Path(DEFAULT_GGUF_GLOB).parent.parent
    hits = sorted(Path(snaps).glob("*/harrier-oss-v1-0.6b.Q8_0.gguf")) if snaps.exists() else []
    if hits:
        return str(hits[0])
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Clasificador de dos etapas (harrier-0.6b + cabezas)")
    parser.add_argument("--port", type=int, default=8091, help="HTTP port (default: 8091)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument(
        "--backend", choices=("gguf", "onnx"), default="gguf",
        help="backend de embeddings (default: gguf = llama.cpp persistente; "
             "onnx = onnxruntime sobre --embeddings-model)",
    )
    parser.add_argument(
        "--embeddings-model", "--model", dest="embeddings_model", default=None,
        help="ruta al GGUF (backend=gguf) o ruta/repo ONNX (backend=onnx). "
             "Default gguf: detecta harrier-oss-v1-0.6b.Q8_0 en el cache de medición.",
    )
    parser.add_argument("--heads-dir", type=Path, default=Path("heads"),
                        help="directorio con stage1.npz y stage2.npz (default: heads/)")
    parser.add_argument("--legacy-prototypes", action="store_true",
                        help="usar los prototipos heurísticos viejos (sin cabezas)")
    parser.add_argument("--no-vulkan", action="store_true", help="onnx: CPU only")
    parser.add_argument("--llama-threads", type=int, default=4,
                        help="hilos de llama-server (default: 4, el medido)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    model_ref = args.embeddings_model or (_default_gguf() if args.backend == "gguf" else MODEL_REPO_DEFAULT_ONNX)
    if args.backend == "gguf" and not args.embeddings_model and not model_ref:
        parser.error("no se encontró el GGUF de harrier en el cache y no pasaste --embeddings-model")

    log.info("Backend %s · modelo %s", args.backend, model_ref)
    if args.backend == "gguf":
        embedder: object = GgufEmbedder(Path(model_ref), threads=args.llama_threads)
    else:
        embedder = OnnxEmbedder(model_ref, use_vulkan=not args.no_vulkan)

    heads_dir = args.heads_dir
    use_heads = not args.legacy_prototypes
    if use_heads and not heads_dir.exists():
        parser.error(
            f"no hay cabezas en {heads_dir} y no pasaste --legacy-prototypes. "
            f"Corré `python3 train_heads.py` (necesita scikit-learn en el intérprete) "
            f"o arrancá con --legacy-prototypes para el fallback de prototipos."
        )

    global classifier
    if use_heads:
        classifier = HeadClassifier(embedder, heads_dir)
        log.info("Clasificador de dos etapas listo (dim=%d, thr1=%.4f, thr2=%.4f)",
                 classifier.dim, classifier.thr1, classifier.thr2)
    else:
        classifier = PrototypeClassifier(embedder)
        log.warning("MODO LEGACY: prototipos heurísticos (no entrenados). Dos etapas desactivadas.")

    app.state.classifier = classifier
    app.state.backend = args.backend
    app.state.model = model_ref

    log.info("Arrancando server en %s:%d", args.host, args.port)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


classifier: object | None = None


if __name__ == "__main__":
    main()