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

from influence_function import s_test, grad_z  
from utils import save_json, display_progress, get_default_config  


def _to_scalar(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().numpy().tolist()
    if isinstance(x, np.ndarray):
        return x.item() if x.ndim == 0 else x.tolist()
    return float(x)


def compute_s_test_vector_for_point(model, z_test, t_test, train_loader, device=-1,
                       damp=0.01, scale=25, recursion_depth=5000, r=1):
    """Compute s_test (inverse-HVP * grad) for one test point."""
    model.to(torch.device('cpu') if device == -1 else torch.device(f'cuda:{device}'))
    s_vec_list=[]
    for i in range(r):
        s_vec=s_test(z_test=z_test, t_test=t_test, model=model, z_loader=train_loader, device=device, damp=damp, scale=scale, recursion_depth=recursion_depth)
        display_progress("Averaging r-times: ", i, r)
        s_flat = torch.cat([s.flatten() for s in s_vec]).to(device)
        s_vec_list.append(s_flat)
    s_avg = torch.stack(s_vec_list).mean(dim=0)
    print(f"Shape of s_vec: {s_avg.shape}")
    return s_avg



def compute_influence_for_test_point(model, train_loader, test_loader, test_index, config, s_vec=None, time_logging=False):
    """Return influences, harmful idxs, helpful idxs for a single test point over entire training set."""
    z_test, t_test = test_loader.dataset[test_index]
    if s_vec is None:
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])
        s_vec = compute_s_test_vector_for_point(model, z_test, t_test, train_loader, device=config['device'], damp=config['damp'], scale=config['scale'], recursion_depth=config['recursion_depth'], r=config['r_averaging'])
    n_train = len(train_loader.dataset)
    influences = []
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    s_vec=s_vec.to(device)
    start=datetime.datetime.now()
    for i in range(n_train):
        z_train, t_train = train_loader.dataset[i]
        z_train = train_loader.collate_fn([z_train])
        t_train = train_loader.collate_fn([t_train])
        if time_logging:
            time_a = datetime.datetime.now()
        grad_vec = grad_z(z_train, t_train, model, device=config['device'])
        grad_vec=[g.to(device) for g in grad_vec]
        influence_val = -sum(torch.sum(g * s).detach().cpu().item()
                             for g, s in zip(grad_vec, s_vec)) / n_train
        if time_logging:
            time_b = datetime.datetime.now()
            time_delta = time_b - time_a
            logging.debug(f"Time for single influence pair in ms:"f" {time_delta.total_seconds() * 1000}")
        influences.append(influence_val)
        display_progress("Calc. influence function: ", i, n_train)
    end=datetime.datetime.now()
    diff=end-start
    logging.info(f"Time for dataset-influence of single test point:"f" {diff.total_seconds() * 1000}")
    harmful = np.argsort(influences).tolist()
    helpful = harmful[::-1]
    return influences, harmful, helpful


def calc_influence_batch(model, train_loader, test_loader, indices=None, config=None, cachedir=None):
    """
    Compute influences for every test index in indices
    """
    if config is None:
        config = get_default_config()

    outdir = Path(config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)
    if cachedir:
        cachedir = Path(cachedir)
        cachedir.mkdir(exist_ok=True, parents=True)

    results = {}
    test_len=0
    if indices is None:
        test_len = len(test_loader.dataset)
        indices=list(range(len(test_loader.dataset)))
    else:
        test_len=len(indices)

    for j in range(test_len):
        start = time.time()
        z_test, t_test = test_loader.dataset[indices[j]]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])
        infl, harmful, helpful = compute_influence_for_test_point(
            model, train_loader, test_loader=test_loader, test_index=indices[j], config=None
        )

        # Convert everything to JSON-friendly types
        infl_json = [_to_scalar(v) for v in infl]
        results[str(j)] = {
            "label": _to_scalar(t_test[0]),
            "num_in_dataset": indices[j],
            "time_calc_influence_s": time.time() - start,
            "influence": infl_json,
            "harmful": [int(x) for x in harmful],
            "helpful": [int(x) for x in helpful],
        }

        tmp_path = outdir / f"influence_results_tmp_last-i_{j}.json"
        save_json(results, tmp_path)
        display_progress("Test samples processed: ", j, test_len)

    final_path = outdir / "influence_results.json"
    save_json(results, final_path)
    logging.info(f"Saved influence results to {final_path}")
    return results


def calc_influence_on_pair(model, train_loader, test_loader, train_id, test_id, s_vec=None, device=-1, damp=0.01, scale=25, recursion_depth=5000, r=10):
    gpu=device
    device = torch.device('cpu') if device == -1 else torch.device(f'cuda:{device}')
    time_a=datetime.datetime.now()
    model.to(device)
    z_train, t_train = train_loader.dataset[train_id]
    z_train = train_loader.collate_fn([z_train]).to(device)
    t_train = train_loader.collate_fn([t_train]).to(device)
    z_test, t_test = train_loader.dataset[test_id]
    z_test = test_loader.collate_fn([z_test]).to(device)
    t_test = test_loader.collate_fn([t_test]).to(device)

    if s_vec is None:
        s_vec = compute_s_test_vector_for_point(model, z_test, t_test, train_loader, device=gpu, damp=damp, scale=scale, recursion_depth=recursion_depth, r=r)
    s_vec = s_vec.to(device)
    grad_vec=grad_z(z_train, t_train, model, device=gpu)
    grad_vec = [g.to(device) for g in grad_vec] if isinstance(grad_vec, list) else grad_vec.to(device)
    influence_val = -sum(torch.sum(g * s).detach().cpu().item() for g, s in zip(grad_vec, s_vec))
    time_b=datetime.datetime.now()
    diff=time_b-time_a
    logging.info("Time for influence of single z_train z-test pair:"f" {(train_id, test_id)}: {diff.total_seconds() * 1000}")
    return influence_val

def calc_average_influence_of_point(model, train_loader, test_loader, train_index, config, test_indices=None):
    """
    Compute the average influence of a single training point across all provided test points.
    Returns a single scalar (sum of influences).
    """
    if test_indices is None:
        test_indices = list(range(len(test_loader.dataset)))
    
    total_influence = 0.0
    for test_idx in test_indices:
        inf=calc_influence_on_pair(model=model, train_loader=train_loader, test_loader=test_loader, train_id=train_index, test_id=test_idx, device=config['device'], damp=config['damp'], scale=config['scale'], recursion_depth=config['recursion_depth'], r=config['r_averaging'])
        total_influence+=inf
    avg_influence = total_influence / len(test_indices)
    return avg_influence


