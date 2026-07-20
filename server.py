"""
Minilm-based intent classifier server for Kateto.

Exposes POST /v1/chat/completions matching the ClassifierProvider contract.
Uses ONNX Runtime with optional GPU acceleration.

Intent classification via embedding similarity:
  - Mean-pooled last_hidden_state as sentence embedding (384-d)
  - Cosine similarity to per-class prototype sentences
  - Returns {category, confidence} in OpenAI-compatible format

Usage:
    uv venv && uv pip sync requirements.txt
    python server.py                          # default http://127.0.0.1:8091
    python server.py --port 9091              # custom port
    python server.py --model path/to/model    # local ONNX model
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

log = logging.getLogger("mmbert-server")

# ---------------------------------------------------------------------------
# Classification labels — must match kateto.core.event.Classification
# ---------------------------------------------------------------------------

CATEGORIES = ("EXECUTE", "IGNORE_SELF_TALK", "IGNORE_THIRD_PARTY")

# Prototype sentences per category — embedded at startup for similarity matching
# ponytail: heuristic prototypes, not trained. Swap for learned embeddings when
# labeled data exists under classifiers/mmbert/training/.
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
# Model loading
# ---------------------------------------------------------------------------

MODEL_REPO = "Qdrant/all-MiniLM-L6-v2-onnx"
ONNX_FILENAME = "model.onnx"
CONFIG_FILENAME = "config.json"


def _resolve_model_path(model_ref: str) -> Path:
    """Return local path to the ONNX model file.

    If *model_ref* looks like a filesystem path, use it directly.
    Otherwise treat it as a HuggingFace Hub repo ID and download.
    """
    p = Path(model_ref)
    if p.exists():
        return p if p.is_file() else p / ONNX_FILENAME
    # Download from HuggingFace Hub
    return Path(
        hf_hub_download(
            repo_id=model_ref,
            filename=ONNX_FILENAME,
            local_files_only=False,
        )
    )


def _load_tokenizer(model_repo: str) -> Tokenizer:
    """Load the BERT-compatible WordPiece tokenizer from the Hub."""
    tokenizer_path = hf_hub_download(
        repo_id=model_repo,
        filename="tokenizer.json",
        local_files_only=False,
    )
    return Tokenizer.from_file(tokenizer_path)


def _create_session(model_path: Path, *, use_vulkan: bool = True) -> ort.InferenceSession:
    """Create an ONNX Runtime session with optional Vulkan GPU provider."""
    providers = []
    if use_vulkan:
        providers.append("VulkanExecutionProvider")
    providers.append("CPUExecutionProvider")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_cpu_mem_arena = True
    opts.enable_mem_pattern = True

    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=providers,
        )
    except Exception as exc:
        if use_vulkan and "Vulkan" in str(exc):
            log.warning("Vulkan provider unavailable (%s), falling back to CPU", exc)
            session = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        else:
            raise

    active = session.get_providers()
    log.info("ONNX Runtime providers: %s", active)
    return session


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _tokenize(tokenizer: Tokenizer, texts: list[str], max_length: int = 128) -> dict[str, np.ndarray]:
    """Tokenize a batch of texts for BERT ONNX input."""
    encoded = tokenizer.encode_batch(texts)
    input_ids = np.zeros((len(texts), max_length), dtype=np.int64)
    attention_mask = np.zeros((len(texts), max_length), dtype=np.int64)
    token_type_ids = np.zeros((len(texts), max_length), dtype=np.int64)

    for i, enc in enumerate(encoded):
        ids = enc.ids[:max_length]
        length = len(ids)
        input_ids[i, :length] = ids
        attention_mask[i, :length] = 1
        # token_type_ids stays zero (single-sentence input)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def _embed(session: ort.InferenceSession, tokenizer: Tokenizer, texts: list[str]) -> np.ndarray:
    """Compute normalised sentence embeddings via mean pooling of last_hidden_state.

    Returns shape (N, 384) — L2-normalised vectors.
    """
    inputs = _tokenize(tokenizer, texts)
    outputs = session.run(None, inputs)
    last_hidden = outputs[0]  # shape (N, seq_len, 384) — last_hidden_state

    # Mean pool — average token embeddings weighted by attention_mask
    mask = inputs["attention_mask"].astype(np.float32)
    mask_expanded = mask[:, :, np.newaxis]  # (N, seq_len, 1)
    summed = np.sum(last_hidden * mask_expanded, axis=1)
    counts = np.maximum(np.sum(mask, axis=1, keepdims=True), 1e-9)
    embeddings = summed / counts  # (N, 384)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # guard against zero vectors
    return embeddings / norms


# ---------------------------------------------------------------------------
# Similarity classifier
# ---------------------------------------------------------------------------

class PrototypeClassifier:
    """Nearest-prototype classifier using cosine similarity of BERT embeddings."""

    def __init__(self, session: ort.InferenceSession, tokenizer: Tokenizer) -> None:
        self.session = session
        self.tokenizer = tokenizer
        self._prototype_embeddings: dict[str, np.ndarray] = {}
        self._build_prototypes()

    def _build_prototypes(self) -> None:
        for cat, sentences in PROTOTYPES.items():
            emb = _embed(self.session, self.tokenizer, sentences)
            class_center = emb.mean(axis=0)  # centroid of per-class prototypes
            class_center /= np.linalg.norm(class_center)  # re-normalise
            self._prototype_embeddings[cat] = class_center
            log.info(
                "prototype %s: %d sentences, centroid norm=%.4f",
                cat,
                len(sentences),
                np.linalg.norm(class_center),
            )

    def classify(self, text: str, agents: list[str] | None = None) -> tuple[str, float]:
        """Return (category, confidence) for *text*.

        If *agents* is provided and any name appears in *text*,
        EXECUTE gets a +0.25 logit boost (addresses a known agent).
        """
        emb = _embed(self.session, self.tokenizer, [text])[0]
        best_cat: str = CATEGORIES[-1]
        best_sim = -1.0

        sims = np.array([
            float(np.dot(emb, self._prototype_embeddings[c]))
            for c in CATEGORIES
        ])

        # Boost EXECUTE when text mentions a known agent name
        if agents:
            text_lower = text.lower()
            if any(name.lower() in text_lower for name in agents):
                sims[CATEGORIES.index("EXECUTE")] += 0.25

        best_idx = int(np.argmax(sims))
        best_cat = CATEGORIES[best_idx]
        best_sim = float(sims[best_idx])

        # Softmax confidence
        sims -= sims.max()
        exp_s = np.exp(sims * 2.0)
        probs = exp_s / exp_s.sum()
        confidence = float(probs[best_idx])

        return best_cat, confidence


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="mmBERT Classifier", version="0.1.0")
classifier: PrototypeClassifier | None = None


@app.on_event("startup")
async def _noop() -> None:
    """Keep ref for lifetime — classifier set in main() before uvicorn.run()."""
    app.state.classifier = classifier


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    """OpenAI-compatible chat completions endpoint.

    Expected request body (matching ClassifierRequest):
      {
        "model": "classifier",
        "messages": [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "the text to classify"}
        ],
        "temperature": 0.0,
        "stream": false,
        "response_format": {"type": "json_object"}
      }

    Returns:
      {
        "choices": [{"message": {"content": "{\\"category\\": \\"EXECUTE\\", \\"confidence\\": 0.95}"}}]
      }
    """
    clf = app.state.classifier
    if clf is None:
        return JSONResponse(
            status_code=503,
            content={"error": "classifier not initialized"},
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid JSON"})

    messages = body.get("messages", [])
    if not messages:
        return JSONResponse(status_code=400, content={"error": "messages required"})

    # Extract the user message text (last user message)
    user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") in ("user", "system"):
            user_text = msg.get("content", "")
            break
    if not user_text:
        return JSONResponse(status_code=400, content={"error": "no user message content"})

    # Optional: list of known agent/speaker names to boost EXECUTE when addressed
    agents = body.get("agents", [])

    category, confidence = clf.classify(user_text, agents=agents)
    payload = json.dumps({"category": category, "confidence": round(confidence, 4)})

    return JSONResponse(content={
        "choices": [
            {
                "message": {"content": payload},
            }
        ],
    })


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok"})


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="mmBERT classifier server")
    parser.add_argument(
        "--port",
        type=int,
        default=8091,
        help="HTTP port (default: 8091, matches kateto classifier endpoint)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--model",
        default=MODEL_REPO,
        help=f"ONNX model path or HF repo ID (default: {MODEL_REPO})",
    )
    parser.add_argument(
        "--no-vulkan",
        action="store_true",
        help="Disable Vulkan GPU provider, use CPU only",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Loading tokenizer from %s ...", MODEL_REPO)
    tokenizer = _load_tokenizer(MODEL_REPO)

    log.info("Loading ONNX model from %s ...", args.model)
    model_path = _resolve_model_path(args.model)
    log.info("Model file: %s (%d MB)", model_path, model_path.stat().st_size // 1024 // 1024)

    session = _create_session(model_path, use_vulkan=not args.no_vulkan)

    log.info("Building prototype embeddings ...")
    global classifier
    classifier = PrototypeClassifier(session, tokenizer)
    log.info("Classifier ready (%d categories, %d dimensions)", len(CATEGORIES), 384)

    log.info("Starting server on %s:%d", args.host, args.port)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
