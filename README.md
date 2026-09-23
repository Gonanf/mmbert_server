# mmbert_server

Clasificador de intención para **Kateto** en dos etapas, sobre embeddings de
`harrier-oss-v1-0.6b` y **cabezas ridge entrenadas** (no prototipos heurísticos).

## Pipeline

```
Etapa 1 — ¿terminó la idea?   wait_asr_incompleto  vs  turno completo
Etapa 2 — ¿requiere respuesta?  (solo si la etapa 1 dice "completo")
           no_response* / no_llenar_silencio  vs  requiere
```

| Veredicto | Categoría | Significado |
|---|---|---|
| `wait` | `WAIT` | el ASR cortó a mitad de frase; hay que esperar más audio |
| `exec` | `EXECUTE` | requiere respuesta |
| `ignore` | `IGNORE_SELF_TALK` / `IGNORE_THIRD_PARTY` | no requiere respuesta; el espejo va a `IGNORE_THIRD_PARTY` si el texto menciona un agente conocido (`agents`) |

Las cabezas se entrenan sobre las etiquetas del banco (ver `train_heads.py`);
cada una es un `RidgeClassifier` lineal con umbral elegido por F1 sobre una
partición interna del train y calibración logística a `proba`. El protocolo
exacto (shuffle semilla fija, split 70/30, pesos de clase) es el mismo del
evaluador del banco (`audit-ridge.py`): los números son reproducibles.

## Endpoints

Todos responden JSON. El server mantiene el contrato OpenAI-compatible que
consume Kateto (`/v1/chat/completions`) y agrega el endpoint explícito.

### `POST /v1/classify`

```bash
curl -s localhost:8091/v1/classify \
  -H 'Content-Type: application/json' \
  -d '{"text": "y después me dijo que le parecía", "context": [], "agents": ["Jane", "Doktor"]}'
```

```json
{"stage1": {"label": "wait", "proba": 0.4133},
 "stage2": {"label": "requiere", "proba": 0.238},
 "verdict": "wait", "category": "WAIT", "confidence": 0.4133}
```

`context` = hasta 10 mensajes previos (se valida; el clasificador decide sobre
`text` solo: las cabezas se entrenaron sobre turnos sueltos).

### `POST /v1/chat/completions`

Mismo shape de respuesta de siempre (lo consume Kateto como provider
OpenAI-compatible):

```bash
curl -s localhost:8091/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "classifier", "messages": [{"role": "user", "content": "y después me dijo que le parecía"}]}'
```

```json
{"choices": [{"message": {"content": "{\"category\": \"WAIT\", \"confidence\": 0.4133}"}}]}
```

### `GET /health`

```json
{"status": "ok", "backend": "gguf", "model": "<ruta>",
 "heads_loaded": true, "mode": "two-stage-heads",
 "latency_p50_ms": 72.4, "n_calls": 5}
```

## Cómo correr

```bash
python3 server.py                          # gguf (default), cabezas de heads/, puerto 8091
python3 server.py --backend onnx --embeddings-model Qdrant/all-MiniLM-L6-v2-onnx
python3 server.py --legacy-prototypes --backend onnx   # fallback de prototipos (explícito)
```

Flags: `--port`, `--host`, `--backend gguf|onnx`, `--embeddings-model <ruta|repo>`
(alias: `--model`), `--heads-dir heads/`, `--legacy-prototypes`, `--no-vulkan`,
`--llama-threads N`, `--log-level DEBUG|INFO|WARNING|ERROR`.

- **gguf** (default): levanta `llama-server` (llama.cpp) en un puerto libre con
  `--pooling mean --embd-normalize 2`, detecta el GGUF Q8_0 de harrier en el
  cache (o pasale la ruta con `--embeddings-model`). El subproceso se mata solo
  al salir.
- **onnx**: onnxruntime; por defecto `Qdrant/all-MiniLM-L6-v2-onnx` (el encoder
  de producción, 384-d). Si el modelo no está cacheado, se descarga con
  huggingface_hub.

Sin cabezas (`heads/stage1.npz`, `heads/stage2.npz`) y sin `--legacy-prototypes`
el server no arranca: error claro.

## Cómo entrenar las cabezas

`train_heads.py` corre con el **python3 del sistema** (tiene scikit-learn; el
venv de entrenamiento no). Reusa el protocolo de `audit-ridge.py` y verifica la
alineación del banco antes de entrenar (falla ruidoso si no coincide).

```bash
python3 train_heads.py   # lee el banco + escenarios; escribe heads/{stage1,stage2}.npz + metrics.json
```

Métricas guardadas en `heads/metrics.json` (acc, recall/precisión de la clase
objetivo, descarte, umbral por etapa).

## Latencia medida

Harrier-0.6b Q8_0 por llama.cpp en este host (4 hilos, servidor persistente):

| Caso | p50 medido |
|---|---|
| Banclo EOT — 2497 turnos (7–70 tokens) | 27–113 ms |
| Smoke real — 5 llamadas mixtas | 72 ms |
| MiniLM-L6 ONNX (encoder de producción) | ~11 ms |

`llama-server` persistente (server levantado) es ~2–10× más lento que el
MiniLM ONNX; el trade-off compra precisión de clasificación (~95 % de
precisión en la clase objetivo de la etapa 2 vs ~77 % del flujo prototipo).
Medido, no estimado: `GET /health` reporta la p50 real de las últimas 100
llamadas.

## Tests

```bash
python3 -m pytest tests/ -q
```

Backend falso determinista (síntetico, sin GPU ni modelo), umbrales aplicados,
mapeo de verdict a categoría, errores claros sin cabezas/dimensiones
incompatibles, contrato de los endpoints, y un test de alineación que falla si
embeddings y etiquetas no coinciden (skip si el banco no está en disco).