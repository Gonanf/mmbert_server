#!/usr/bin/env python3
"""Entrena las dos cabezas ridge del clasificador de dos etapas.

Reusa TAL CUAL el protocolo de `kateto-medicion/banco-eot/audit-ridge.py` (probar()).
El script de medición lo exige: si cada etapa sale de un script distinto con otro
split, la métrica no significa nada.

Protocolo (idéntico a audit-ridge.py):
  - shuffle con semilla fija 42, split 70/30
  - RidgeClassifier(alpha=1.0, class_weight={0:1.0, 1: neg/pos})
  - umbral elegido por F1 sobre una validación interna (20 % del train, semilla 43),
    NUNCA sobre el test

Etapas:
  - stage1: ¿terminó la idea?  clase objetivo = wait_asr_incompleto
  - stage2: ¿requiere respuesta?  clase objetivo = no_response_* / no_llenar_silencio

Además guarda calibración logística (sobre la misma val interna) para convertir la
decisión lineal a proba en servida.

Interprete: `python3` del sistema (tiene scikit-learn; el venv de kateto-train NO).

Salida: heads/stage1.npz, heads/stage2.npz, heads/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier

SEED = 42

NO_TERMINO = {"wait_asr_incompleto"}
NO_REQUIERE = {"no_response_al_aire", "no_response_otra_voz", "no_llenar_silencio"}
REQUIERE = {"solo_charla", "si_responde_indirecto", "silencio_charla_cerrada"}

DEFAULT_SCENARIOS = Path("/run/media/chaos/terciario/proyectos/kateto-train/out/eot-embeddings")
DEFAULT_EMBEDDINGS = Path("/run/media/chaos/terciario/proyectos/kateto-medicion/banco-eot/out/harrier-0.6b.npy")
DEFAULT_OUT = Path(__file__).parent / "heads"


def cargar_escenarios(path: Path) -> list[str]:
    raw = json.load(open(path))
    if isinstance(raw, dict):
        if all(isinstance(v, list) for v in raw.values()):
            esc = [None] * 2497
            for k, idxs in raw.items():
                for i in idxs:
                    esc[i] = k
        else:
            esc = [raw[str(i)] for i in range(2497)]
        return esc
    return list(raw)


def normalizar(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def probar(X: np.ndarray, y: np.ndarray, nombre: str) -> dict:
    """Ridge ponderado; umbral por F1 en validación interna (20 % del train).

    Copia literal de audit-ridge.py::probar + calibración logística sobre la val.
    """
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    corte = int(len(idx) * 0.7)
    tr, te = idx[:corte], idx[corte:]
    # validación interna para el umbral
    rng2 = np.random.default_rng(SEED + 1)
    v = rng2.permutation(len(tr))
    n_val = int(len(v) * 0.2)
    val, tr_eff = tr[v[:n_val]], tr[v[n_val:]]

    pos, neg = (y[tr_eff] == 1).sum(), (y[tr_eff] == 0).sum()
    clf = RidgeClassifier(alpha=1.0, class_weight={0: 1.0, 1: float(neg / max(pos, 1))})
    clf.fit(X[tr_eff], y[tr_eff])
    dv = clf.decision_function(X[val])
    yv = y[val]

    mejor, thr = -1, 0.0
    for t in np.linspace(dv.min(), dv.max(), 200):
        pred = (dv >= t).astype(int)
        tp = int(((pred == 1) & (yv == 1)).sum())
        fp = int(((pred == 1) & (yv == 0)).sum())
        fn = int(((pred == 0) & (yv == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > mejor:
            mejor, thr = f1, t

    dt = clf.decision_function(X[te])
    yt = y[te]
    pred = (dt >= thr).astype(int)
    acc = float((pred == yt).mean())
    tp = int(((pred == 1) & (yt == 1)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    tn = int(((pred == 0) & (yt == 0)).sum())
    recall = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    descarte = tn / max(tn + fp, 1)

    # Calibración logística sobre la misma val interna: decision -> proba
    cal = LogisticRegression()
    cal.fit(dv.reshape(-1, 1), yv)

    return dict(
        nombre=nombre,
        n_train=len(tr_eff),
        n_test=len(te),
        positivos_test=int(yt.sum()),
        acc=acc,
        recall_obj=recall,
        precision_obj=prec,
        descarte=descarte,
        umbral=float(thr),
        f1=float(2 * prec * recall / max(prec + recall, 1e-9)),
        coef=clf.coef_[0].astype(np.float32),
        intercept=float(clf.intercept_[0]),
        cal_w=float(cal.coef_[0][0]),
        cal_b=float(cal.intercept_[0]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    ap.add_argument("--escenarios-dir", type=Path, default=DEFAULT_SCENARIOS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    # ── Carga + alineación (falla ruidoso si no coincide) ──────────────
    esc_path = args.escenarios_dir / "escenarios.json"
    textos_path = args.escenarios_dir / "textos.npy"
    for p in (args.embeddings, esc_path, textos_path):
        if not p.exists():
            sys.exit(f"FALTA archivo: {p}. No entreno sin datos reales.")

    esc = cargar_escenarios(esc_path)
    textos = np.load(textos_path, allow_pickle=True)
    X = np.load(args.embeddings)

    print(f"escenarios: {len(esc)} · textos: {len(textos)} · embeddings: {X.shape}")
    if not (len(textos) == len(esc) == X.shape[0] == 2497):
        sys.exit(f"ALINEACION ROTA: textos={len(textos)} esc={len(esc)} X={X.shape[0]} — esperado 2497.")
    if X.ndim != 2 or X.shape[1] != 1024:
        sys.exit(f"embeddings inesperados: shape {X.shape}, se espera (2497, 1024) de harrier-0.6b.")

    Xn = normalizar(X.astype(np.float64))

    # ── ETAPA 1: todas las muestras ─────────────────────────────────────
    y1 = np.array([1 if e in NO_TERMINO else 0 for e in esc])
    print(f"\n=== ETAPA 1: {len(y1)} muestras "
          f"({int(y1.sum())} objetivo wait_asr_incompleto) ===")
    r1 = probar(Xn, y1, "stage1_wait_asr_incompleto")
    print(f"acc={r1['acc']*100:.2f}% recall_obj={r1['recall_obj']*100:.2f}% "
          f"prec_obj={r1['precision_obj']*100:.2f}% descarte={r1['descarte']*100:.2f}% "
          f"umbral={r1['umbral']:.5f}")

    # ── ETAPA 2: subconjunto no_requiere/requiere ───────────────────────
    idx2 = [i for i, e in enumerate(esc) if e in NO_REQUIERE or e in REQUIERE]
    y2 = np.array([1 if esc[i] in NO_REQUIERE else 0 for i in idx2])
    print(f"\n=== ETAPA 2: {len(idx2)} muestras "
          f"({int(y2.sum())} objetivo no_requiere) ===")
    r2 = probar(Xn[idx2], y2, "stage2_no_requiere")
    print(f"acc={r2['acc']*100:.2f}% recall_obj={r2['recall_obj']*100:.2f}% "
          f"prec_obj={r2['precision_obj']*100:.2f}% descarte={r2['descarte']*100:.2f}% "
          f"umbral={r2['umbral']:.5f}")

    # ── Guardar ─────────────────────────────────────────────────────────
    args.out.mkdir(parents=True, exist_ok=True)
    for tag, r in (("stage1", r1), ("stage2", r2)):
        np.savez(
            args.out / f"{tag}.npz",
            coef=r["coef"], intercept=r["intercept"],
            thr=r["umbral"], cal_w=r["cal_w"], cal_b=r["cal_b"],
            dim=np.int32(X.shape[1]), nombre=r["nombre"],
        )

    metrics = {
        "protocolo": "banco-eot/audit-ridge.py (seed 42, split 70/30, umbral F1 en val interna 20% del train)",
        "embeddings": str(args.embeddings),
        "semilla": SEED,
        "stage1": {k: r1[k] for k in ("nombre", "n_train", "n_test", "positivos_test",
                                      "acc", "recall_obj", "precision_obj", "descarte", "umbral", "f1")},
        "stage2": {k: r2[k] for k in ("nombre", "n_train", "n_test", "positivos_test",
                                      "acc", "recall_obj", "precision_obj", "descarte", "umbral", "f1")},
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nGuardado: {args.out}/stage1.npz, stage2.npz, metrics.json")


if __name__ == "__main__":
    main()