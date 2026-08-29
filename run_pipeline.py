#!/usr/bin/env python3
"""
CLI entry point for the GP Uncertainty / Novel Defect Detection pipeline.

Usage examples:
    python run_pipeline.py                         # all phases, pca_dim=16
    python run_pipeline.py --phases 0 1 2          # build manifest + folds + embeddings
    python run_pipeline.py --phases 3 --pca-dim 8  # GP only, d=8
    python run_pipeline.py --phases 4 5 6          # scoring, baselines, evaluation
    python run_pipeline.py --force                 # recompute everything from scratch
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GP Uncertainty for Novel Defect Detection — MVTec AD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phases", nargs="+", type=int, default=None, metavar="N",
        help="Pipeline phases to run (0–6). Default: all.",
    )
    parser.add_argument(
        "--pca-dim", type=int, default=16, choices=[8, 12, 16],
        help="PCA latent dimension for GP input (default: 16).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device ('cuda', 'cpu'). Default: auto-detect.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore cached results and recompute.",
    )
    args = parser.parse_args()
    run_pipeline(
        phases=args.phases,
        pca_dim=args.pca_dim,
        device=args.device,
        force=args.force,
    )


if __name__ == "__main__":
    main()
