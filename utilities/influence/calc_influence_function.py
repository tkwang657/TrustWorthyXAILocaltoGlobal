#! /usr/bin/env python3
"""
Lightweight influence functions for tabular Torch models (e.g., TabNet).
Designed for small/medium datasets (computes influences on the fly).
"""

import os
import sys
import time
import datetime
import logging
from pathlib import Path

import numpy as np
import torch

# Local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
influence_path = os.path.join(project_root, 'utilities', 'influence')
if influence_path not in sys.path:
    sys.path.append(influence_path)

from influence_function import s_test, grad_z  # noqa: E402
from utils import save_json, display_progress  # noqa: E402


def get_default_config():
    """Config tuned for tabular; keep recursion_depth modest for speed."""
    return {
        'gpu': 0 if torch.cuda.is_available() else -1,
        'damp': 0.01,
        'scale': 25.0,
        'recursion_depth': 300,   # reduce if slow, increase for accuracy
        'r_averaging': 1,
        'top_k': 20,
        'outdir': 'outdir',
    }


def _to_scalar(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().numpy().tolist()
    if isinstance(x, np.ndarray):
        return x.item() if x.ndim == 0 else x.tolist()
    return float(x)


def compute_s_test_for_point(model, z_test, t_test, train_loader, config):
    """Compute s_test (inverse-HVP * grad) for one test point."""
    return s_test(
        z_test,
        t_test,
        model,
        train_loader,
        gpu=config['gpu'],
        damp=config['damp'],
        scale=config['scale'],
        recursion_depth=config['recursion_depth'],
    )


def compute_influence_for_test_point(model, train_loader, z_test, t_test, config):
    """Return influences, harmful idxs, helpful idxs for a single test point."""
    s_vec = compute_s_test_for_point(model, z_test, t_test, train_loader, config)
    n_train = len(train_loader.dataset)
    influences = []

    for i in range(n_train):
        z_train, t_train = train_loader.dataset[i]
        z_train = train_loader.collate_fn([z_train])
        t_train = train_loader.collate_fn([t_train])

        grad_vec = grad_z(z_train, t_train, model, gpu=config['gpu'])
        influence_val = -sum(torch.sum(g * s).detach().cpu().item()
                             for g, s in zip(grad_vec, s_vec)) / n_train
        influences.append(influence_val)
        display_progress("Calc. influence function: ", i, n_train)

    harmful = np.argsort(influences).tolist()
    helpful = harmful[::-1]
    return influences, harmful, helpful


def calc_influence_dataset(model, train_loader, test_loader, config=None):
    """
    Compute influences for every test sample (or the first `config.top_k`
    if you wish to slice afterwards in the notebook for speed).
    """
    if config is None:
        config = get_default_config()

    outdir = Path(config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)

    results = {}
    test_len = len(test_loader.dataset)
    for j in range(test_len):
        start = time.time()
        z_test, t_test = test_loader.dataset[j]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])

        infl, harmful, helpful = compute_influence_for_test_point(
            model, train_loader, z_test, t_test, config
        )

        # Convert everything to JSON-friendly types
        infl_json = [_to_scalar(v) for v in infl]
        results[str(j)] = {
            "label": _to_scalar(t_test[0]),
            "num_in_dataset": j,
            "time_calc_influence_s": time.time() - start,
            "influence": infl_json,
            "harmful": [int(x) for x in harmful[:config['top_k']]],
            "helpful": [int(x) for x in helpful[:config['top_k']]],
        }

        tmp_path = outdir / f"influence_results_tmp_last-i_{j}.json"
        save_json(results, tmp_path)
        display_progress("Test samples processed: ", j, test_len)

    final_path = outdir / "influence_results.json"
    save_json(results, final_path)
    logging.info(f"Saved influence results to {final_path}")
    return results