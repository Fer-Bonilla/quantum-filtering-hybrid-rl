"""Campaña de la máscara-oráculo (v7 — experimento 4: techo del canal selector).

Entrena el PPO con una máscara que, en cada step, contiene los ``m`` activos de
MAYOR retorno futuro (``env._return_matrix[t+1]`` — el mismo que da la
recompensa). Fuga ex-post deliberada y etiquetada. Acota el techo del canal
selector: si ni con la máscara perfecta sube el Sharpe frente al Modelo A, el
cuello de botella es la política, no la selección.

Todo lo demás (PPO, env v2 log_wealth, 10 semillas) es idéntico a las campañas
A/R/Q, de modo que oracle vs A vs R es comparable pareado por semilla. El
oráculo IGNORA el subgrafo (elige global top-m), por lo que evita el quirk de
grafo obsoleto del resto de selectores en evaluación.

Uso::

    uv run python scripts/run_oracle_campaign_v7.py \\
        --universe nivel2 --steps 50000 \\
        --seeds 42 123 456 789 1024 7 99 314 1729 65535
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys as _sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.utils.logging import get_logger, setup_logging
from src.utils.paths import TABLES_DIR, ensure_dir

_path = Path(__file__).resolve().parent / "run_campaign.py"
_spec = importlib.util.spec_from_file_location("run_campaign", _path)
assert _spec is not None and _spec.loader is not None
run_campaign = importlib.util.module_from_spec(_spec)
_sys.modules["run_campaign"] = run_campaign
_spec.loader.exec_module(run_campaign)

_log = get_logger(__name__)


def _run_one(*, seed: int, universe: str, steps: int) -> dict:
    """Un run del PPO con la máscara-oráculo parcheada como step_hook."""
    original_hook = run_campaign._build_hybrid_hook

    def patched_hook(cfg, env, features, classical, model):
        from src.agents.oracle_mask import top_m_future_mask

        m = cfg.quantum.m_top if cfg.quantum is not None else 3
        # Retorno del SIGUIENTE step, alineado con env.current_t y con el orden
        # de acciones (tickers ordenados). Es el mismo que determina la reward.
        returns = np.asarray(env._return_matrix)
        n = env.n_tickers

        def hook(e, obs, info):
            t = int(e.current_t)
            if t + 1 >= returns.shape[0]:
                return np.ones(n, dtype=bool)  # sin futuro -> all-True
            return top_m_future_mask(returns[t + 1], m)

        return hook

    run_campaign._build_hybrid_hook = patched_hook
    try:
        row = run_campaign._train_and_evaluate_one(
            model="D",  # base con graph/quantum presentes; el hook se sustituye
            seed=seed,
            universe=universe,
            steps=steps,
            campaign_id="oracle_v7",
            config_suffix="v2",
        )
    finally:
        run_campaign._build_hybrid_hook = original_hook

    d = asdict(row)
    d["model"] = "ORACLE"
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="nivel2")
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[42, 123, 456, 789, 1024, 7, 99, 314, 1729, 65535],
    )
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    run_campaign._maybe_download(args.universe)
    output = args.output or TABLES_DIR / "oracle_v7.csv"
    ensure_dir(output.parent)

    rows: list[dict] = []
    # Reanudación idempotente (como el barrido de rotación).
    done: set[int] = set()
    if output.exists():
        existing = list(csv.DictReader(output.open("r", encoding="utf-8")))
        rows.extend(existing)
        done = {int(r["seed"]) for r in existing}
        print(f"[resume] {len(existing)} runs ya presentes en {output}", flush=True)

    for i, seed in enumerate(args.seeds, 1):
        if seed in done:
            print(f"[{i}/{len(args.seeds)}] seed={seed} (ya hecho, skip)", flush=True)
            continue
        print(f"[{i}/{len(args.seeds)}] ORACLE seed={seed}", flush=True)
        row = _run_one(seed=seed, universe=args.universe, steps=args.steps)
        rows.append(row)
        print(
            f"  sharpe={row['sharpe_ratio']:+.4f} "
            f"cand_hit={row['candidate_hit_rate']:.3f} "
            f"topm={row['topm_hit_rate']:.3f} "
            f"cumret={row['cumulative_return']:+.3f} "
            f"dur={row['duration_seconds']:.1f}s",
            flush=True,
        )
        with output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n[OK] {len(rows)} runs en {output}", flush=True)


if __name__ == "__main__":
    main()
