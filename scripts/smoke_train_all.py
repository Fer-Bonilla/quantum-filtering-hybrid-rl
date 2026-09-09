"""Smoke test final: descarga datos reales y entrena A→B→C→D brevemente.

Verifica que el CLI completo funcione contra datos reales del S&P 500.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from src.data.download import fetch_ohlcv
from src.utils.logging import setup_logging


def main() -> None:
    setup_logging("WARNING")  # silenciar info messages para ver solo lo importante

    # Universo pequeño: 6 tickers, 18 meses
    tickers = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "KO"]
    start = date(2023, 1, 3)
    end = date(2024, 6, 30)
    print(f"\n[1/2] Descargando {len(tickers)} tickers ({start}..{end})...")
    fetch_ohlcv(tickers, start, end, universe="smoke_train", interval="1d")

    # Crear config inline (sin tocar disco)
    import yaml

    cfg_dir = Path("configs/experiment")
    cfg_smoke = {
        "data": {
            "tickers": tickers,
            "start_date": str(start),
            "end_date": str(end),
            "source": "yahoo",
            "interval": "1d",
            "cache_subdir": "smoke_train",
            "auto_adjust": True,
        },
        "env": {
            "feature": {
                "window_length": 10,
                "volatility_window": 10,
                "volume_window": 10,
                "return_type": "log",
                "indicators": [],
            },
            "split": {"train_frac": 0.6, "val_frac": 0.2},
            "lambda_risk": 0.1,
            "mu_cost": 0.001,
            "transaction_cost": 0.0005,
            "max_steps": 50,
            "episode_sampling": "random",
        },
        "agent": {
            "ppo": {
                "learning_rate": 3.0e-4,
                "n_epochs": 2,
                "batch_size": 32,
                "rollout_steps": 128,
                "clip_coef": 0.2,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "entropy_coef": 0.01,
                "value_coef": 0.5,
                "max_grad_norm": 0.5,
                "hidden_sizes": [64, 64],
            },
        },
        "graph": {
            "alpha": 1.0,
            "beta": 0.0,
            "eps": 1.0e-8,
            "lookback_window": 30,
            "k_neighbors": 2,
            "sym_mode": "avg",
            "subgraph_max_size": 4,
            "seed_score_window": 10,
            "update_frequency": 1,
        },
        "quantum": {
            "k_steps": 3,
            "m_top": 2,
            "backend": "matrix",
            "shots": None,
            "init_mode": "uniform",
            "renormalize_threshold": 1.0e-9,
            "noise": {"enabled": False, "depolarizing_prob": 0.0, "dephasing_prob": 0.0},
        },
        "training": {
            "total_steps": 256,
            "eval_every": 128,
            "snapshot_every": 256,
            "mlflow_experiment_name": "smoke",
            "mlflow_tracking_uri": None,
        },
        "seeds": [42],
    }

    for model in ("A", "B", "C", "D"):
        cfg = dict(cfg_smoke)
        cfg["agent"] = {**cfg_smoke["agent"], "model": model}
        # A no usa graph ni quantum
        if model == "A":
            cfg.pop("graph", None)
            cfg.pop("quantum", None)
        elif model == "B":
            cfg.pop("quantum", None)
        cfg["training"] = {**cfg_smoke["training"], "mlflow_experiment_name": f"smoke_{model}"}
        path = cfg_dir / f"smoke_{model}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    print("\n[2/2] Entrenando A, B, C, D (256 steps cada uno)...")
    import subprocess

    for model in ("A", "B", "C", "D"):
        print(f"\n--- Modelo {model} ---")
        cmd = [
            sys.executable,
            "-m",
            "src.main",
            "train",
            "--config",
            str(cfg_dir / f"smoke_{model}.yaml"),
            "--seed",
            "42",
            "--steps",
            "256",
        ]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FAIL {model}:")
            print(result.stdout)
            print(result.stderr)
            sys.exit(result.returncode)
        # Mostrar la última línea con métricas
        for line in result.stderr.splitlines()[-3:]:
            print(line)

    print("\n[OK] Los 4 modelos completaron sin error.\n")


if __name__ == "__main__":
    main()
