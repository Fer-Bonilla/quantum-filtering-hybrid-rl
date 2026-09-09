"""EXP-2 v8 — Contenido informacional de la máscara (Tier 1, exploratorio).

Información mutua I(máscara; pertenencia futura al quintil prometedor) por
selector, sobre los pares (paso, activo) agrupados del replay del EXP-1
(``mask_steps_v8.npz``). Estimador plug-in con corrección de Miller–Madow.
Umbral de significancia por permutación: 1 000 réplicas con desplazamiento
circular temporal de la matriz ``promising`` (offset uniforme en
[h+1, T−h−1]), según el pre-registro v8 (commit 0e3ee5e). IC por bootstrap
sobre semillas.

Uso::

    uv run python scripts/run_mask_information_v8.py
"""

from __future__ import annotations

import csv

import numpy as np

from src.utils.paths import TABLES_DIR

NPZ = TABLES_DIR / "mask_steps_v8.npz"
OUT = TABLES_DIR / "mask_information_v8.csv"
SELECTORS = ["D", "C", "R", "Q", "S0_0", "S0_25", "S0_5", "S1_0"]
N_PERM = 1000
B_BOOT = 5000
H = 5  # horizonte del criterio prometedor


def _mi_plugin(x: np.ndarray, y: np.ndarray) -> float:
    """MI binaria plug-in (nats) con corrección de Miller–Madow."""
    n = len(x)
    if n == 0:
        return float("nan")
    mi = 0.0
    cells = 0
    for xv in (0, 1):
        px = np.mean(x == xv)
        for yv in (0, 1):
            pxy = np.mean((x == xv) & (y == yv))
            py = np.mean(y == yv)
            if pxy > 0 and px > 0 and py > 0:
                mi += pxy * np.log(pxy / (px * py))
                cells += 1
    # Miller–Madow: sesgo ~ (celdas_no_vacias - |X| - |Y| + 1) / (2n)
    correction = max(cells - 2 - 2 + 1, 0) / (2 * n)
    return float(mi + correction)


def _pairs(masks: np.ndarray, ts: np.ndarray, promising: np.ndarray,
           t_shift: int = 0):
    """Vectores X (pertenencia a máscara) e Y (prometedor) agrupados."""
    T = promising.shape[0]
    ts_eff = (ts + t_shift) % T
    y = promising[ts_eff]                     # (steps, N)
    x = masks                                  # (steps, N)
    return x.ravel(), y.ravel()


def main() -> None:
    data = np.load(NPZ)
    promising = data["promising"].astype(bool)
    T = promising.shape[0]
    rng = np.random.default_rng(20260705)

    rows = []
    print(f"EXP-2: MI(mascara; prometedor) — {N_PERM} permutaciones, "
          f"bootstrap B={B_BOOT}\n")
    for sel in SELECTORS:
        masks = data[f"{sel}__masks"].astype(bool)
        ts = data[f"{sel}__t"]
        seeds = data[f"{sel}__seed"]
        x, y = _pairs(masks, ts, promising)
        mi_obs = _mi_plugin(x, y)

        # Null por permutación circular temporal
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            off = int(rng.integers(H + 1, T - H - 1))
            xn, yn = _pairs(masks, ts, promising, t_shift=off)
            null[i] = _mi_plugin(xn, yn)
        p_perm = float((null >= mi_obs).mean())

        # IC por bootstrap sobre semillas
        uniq = np.unique(seeds)
        mis = []
        for s in uniq:
            sel_idx = seeds == s
            xs, ys = _pairs(masks[sel_idx], ts[sel_idx], promising)
            mis.append(_mi_plugin(xs, ys))
        mis = np.array(mis)
        boots = np.array([rng.choice(mis, size=len(mis), replace=True).mean()
                          for _ in range(B_BOOT)])
        rows.append({
            "selector": sel, "mi_nats": mi_obs,
            "mi_null_mean": float(null.mean()),
            "p_perm": p_perm,
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)),
            "n_pairs": int(len(x)),
        })
        print(f"  {sel:<6} MI={mi_obs:.6f} nats  null={null.mean():.6f}  "
              f"p_perm={p_perm:.4f}  IC95[{rows[-1]['ci_lo']:.6f},"
              f"{rows[-1]['ci_hi']:.6f}]")

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] {OUT}")

    d = next(r for r in rows if r["selector"] == "D")
    r_ = next(r for r in rows if r["selector"] == "R")
    print(f"\nLectura: I_D={d['mi_nats']:.6f} vs I_R={r_['mi_nats']:.6f} "
          f"(criterio pre-registrado: I_D ~ I_R => equivalencia explicada a "
          f"nivel de senal; I_D > I_R => apoya Escenario B)")


if __name__ == "__main__":
    main()
