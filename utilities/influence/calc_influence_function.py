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
from utils import save_json, display_progress, get_default_config, format_time


def _to_scalar(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().numpy().tolist()
    if isinstance(x, np.ndarray):
        return x.item() if x.ndim == 0 else x.tolist()
    return float(x)


def compute_s_test_vector_for_point(model, z_test, t_test, train_loader, device=-1,
                       damp=0.01, scale=25, recursion_depth=5000, r=1, eps=2e-4, patience=None):
    """Compute s_test (inverse-HVP * grad) for one test point."""
    model.to(torch.device('cpu') if device == -1 else torch.device(f'cuda:{config['device']}'))
    s_vec_list=[]
    for i in range(r):
        s_vec=s_test(z_test=z_test, t_test=t_test, model=model, z_loader=train_loader, device=device, damp=damp, scale=scale, recursion_depth=recursion_depth, eps=eps, patience=patience)

        s_flat = torch.cat([s.flatten() for s in s_vec]).to(device)
        s_vec_list.append(s_flat)
        display_progress("Averaging r-times: ", i, r)
    s_avg = torch.stack(s_vec_list).mean(dim=0)
    # s_vec_list=[j.cpu().numpy() for j in s_vec_list]
    # for i in range(1, r):
    #     diff = np.linalg.norm(s_vec_list[i] - s_vec_list[i-1]) / np.linalg.norm(s_vec_list[i])
    #     print(f"normed difference between round {i-1} ({np.linalg.norm(s_vec_list[i])}) and {i} (({np.linalg.norm(s_vec_list[i])}): {diff}")
    return s_avg



def compute_influence_for_test_point(model, train_loader, test_loader, test_index, config, train_ids=None, s_vec=None, time_logging=False):
    """Return influences, harmful idxs, helpful idxs for a single test point over entire training set. influences is a dict that stores influence by the key influences[test_id][train_id]"""
    z_test, t_test = test_loader.dataset[test_index]
    if s_vec is None:
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])
        s_vec = compute_s_test_vector_for_point(model, z_test, t_test, train_loader, device=config['device'], damp=config['damp'], scale=config['scale'], recursion_depth=config['recursion_depth'], r=config['r_averaging'])
    n_train=0
    if train_ids is None:
        n_train = len(train_loader.dataset)
    else:
        n_train=len(train_ids)
    influences = {}
    influences[test_index]={}
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    s_vec=s_vec.to(device)
    start=datetime.datetime.now()
    for i in range(n_train):
        idx=train_ids[i]
        z_train, t_train = train_loader.dataset[idx]
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
            logging.info(f"Time for gradz in seconds is:"f" {time_delta.total_seconds()}")
        influences[test_index][idx]=influence_val
        display_progress("Calc. influence function: ", i, n_train)
    end=datetime.datetime.now()
    diff=end-start
    logging.info(f"Time for dataset-influence of single test point:"f" {diff.total_seconds()}")
    harmful = np.argsort(influences).tolist()
    helpful = harmful[::-1]
    return influences, harmful, helpful


def load_stest_and_compute_batch_influence(model, train_loader, test_loader, test_indices=None, train_indices=None, config=None, cachedir=None, batchsize=512):
    """
    Compute influences for every test index in indices
    """

    outdir = Path(config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)
    outfile=outdir /"influences.csv"
    
    n_train=len(train_indices)
    n_test=len(test_indices)
    svec_cache={}
    batchcount=0
    
    devicename=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device=config['device']
    starttime=datetime.datetime.now()
    computed_pairs = set()
    if outfile.exists():
        with open(outfile, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                train_id, test_id, _ = row
                computed_pairs.add((int(train_id), int(test_id)))
    with open(outfile, "w") as f:
        f.write("train_id,test_id,influence\n")
        for start_idx in range(0, n_train, batchsize):
            time_a=datetime.datetime.now()
            batchcount+=1
            end_idx = min(start_idx + batchsize, n_train)
            batch_indices = [train_indices[i] for i in range(start_idx, end_idx)]
            
            grads_flat_batch = []
            for train_idx in batch_indices:
                z_train, t_train = train_loader.dataset[train_idx]
                z_train = train_loader.collate_fn([z_train]).to(device)
                t_train = train_loader.collate_fn([t_train]).to(device)
                grad_vec = grad_z(z_train, t_train, model, device=device)
                grad_flat = torch.cat([g.flatten() for g in grad_vec]).to(device)
                grads_flat_batch.append(grad_flat)
                
            # Stack batch into matrix [batch_size x num_params]
            grads_batch = torch.stack(grads_flat_batch)  # shape: (batch, num_params)
            tqdm.write(f"Grads for batch {batchcount}/{(n_train//batchsize+1)} done")
            
            for j in range(n_test):
                test_idx=test_indices[j]
                if test_idx in svec_cache:
                    s_vec_flat=svec_cache[test_idx]
                else:
                    timerstart=datetime.datetime.now()
                    # Load precomputed s_test vector
                    s_path = os.path.join(cachedir, f"s_test_{test_idx}.pt")
                    assert os.path.isfile(s_path), f"File path given: {s_path}"
                    s_vec = torch.load(s_path, map_location=devicename)
                    s_vec_flat = torch.cat([p.flatten() for p in s_vec]).to(device)  # shape: (num_params,)
                    svec_cache[test_idx]=s_vec_flat
                # Vectorized influence calculation: - grads @ s_test / n_train
                influence_vals = -(grads_batch @ s_vec_flat) / n_train  # shape: (batch,)
                influence_vals = influence_vals.detach().cpu().numpy()  # convert to numpy only once
                assert influence_vals.shape[0] == grads_batch.shape[0], f"Expected influence_vals length {grads_batch.shape[0]}, got {influence_vals.shape[0]}"
                print(f"Influence_vals shape: {influence_vals.shape()}, {influence_vals}")
                for train_idx, infl in zip(batch_indices, influence_vals):
                    if (train_idx, test_idx) in computed_pairs:
                        continue
                    f.write(f"{train_idx},{test_idx},{infl}\n")
            timerend=datetime.datetime.now()
            elapsed=timerend-starttime
            time_taken=timerend-time_a
            remaining=(elapsed / batchcount) * ((n_train//batchsize +1) - batchcount)
            display_progress(f"Influence computed for Batch {batchcount}", batchcount, n_train//batchsize +1)
            tqdm.write(f"Time Taken: {format_time(str(time_taken.total_seconds()))} | Elapsed: {format_time(str(elapsed.total_seconds()))} | ETA: {format_time(str(remaining.total_seconds()))}")
    loggin.info("Influence Computation complete")
    return influence_dict


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
        s_vec = compute_s_test_vector_for_point(model, z_test, t_test, train_loader, device=gpu, damp=damp, scale=scale, recursion_depth=recursion_depth, r=r, patience=5, eps=5e-4)
    s_vec = s_vec.to(device)
    #logging.info(f"S_vec for test point {test_id}: {s_vec}}")
    grad_vec=grad_z(z_train, t_train, model, device=gpu)
    grad_vec = [g.to(device) for g in grad_vec] if isinstance(grad_vec, list) else grad_vec.to(device)
    influence_val = -sum(torch.sum(g * s).detach().cpu().item() for g, s in zip(grad_vec, s_vec))
    time_b=datetime.datetime.now()
    diff=time_b-time_a
    logging.info("Influence of single z_train z-test pair:"f" {(train_id, test_id)} : {influence_val}. Time: {diff.total_seconds()} seconds")
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

import json
from tqdm import tqdm
def precompute_s_tests(model, test_loader, train_loader, test_ids, config, out_dir="s_test_cache", eps=2e-4, patience=None):
    """
    Precompute s_test vectors for *every* test example and save them individually.
    - Uses streaming saves to avoid loading everything in RAM.
    - Supports FP16 to reduce disk usage (optional).
    - Auto-resumes if some s_test files already generated.
    """
    devicename=torch.device('cpu') if config['device'] == -1 else torch.device(f'cuda:{config["device"]}')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    s_meta = {
        "recursion_depth": config['recursion_depth'],
        "r_avg": config['r_averaging'],
        "damping": config['damp'],
        "scale": config['scale'],
        "num_params": sum(p.numel() for p in model.parameters()),
    }
    json.dump(s_meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    points=len(test_ids)
    start_time=datetime.datetime.now()
    for i in range(points):
        timerstart=datetime.datetime.now()
        idx=test_ids[i]
        out_path = os.path.join(out_dir, f"s_test_{idx}.pt")
        # Skip if already computed (smart resume)
        if os.path.isfile(out_path):
            tqdm.write(f"[skip] s_test for idx {idx} already exists.")
            continue

        logging.info(f"[compute] s_test for idx {idx}")
        z_test, t_test = train_loader.dataset[idx]
        z_test = test_loader.collate_fn([z_test]).to(devicename)
        t_test = test_loader.collate_fn([t_test]).to(devicename)
        s_test_single=compute_s_test_vector_for_point(model=model, z_test=z_test, t_test=t_test, train_loader=train_loader, device=config['device'], damp=config['damp'], scale=config['scale'], recursion_depth=config['recursion_depth'], r=config['r_averaging'], eps=eps, patience=patience)
        # Save a single s_test vector → extremely safe for large param count (131k+)
        print(f"All finite: {torch.isfinite(s_test_single).any().item()}")
        torch.save(s_test_single, out_path)
        timer_end=datetime.datetime.now()
        time_taken=timer_end-timerstart
        elapsed = timer_end - start_time
        completed = i + 1
        remaining = (elapsed / completed) * (points - completed)
        tqdm.write(f"Time for this s_test: {format_time(str(time_taken.total_seconds()))} | "
                   f"Elapsed: {format_time(str(elapsed.total_seconds()))} | ETA: {format_time(str(remaining.total_seconds()))}")
        display_progress("Precomputing s_test", i, points, enabled=True, fix_zero_start=False)
    print(f"Completed s_test precomputation. Saved to {out_dir}/")


