#! /usr/bin/env python3

import torch
from torch.autograd import grad
from utils import display_progress
import sys
import logging as logging
import os
cur=os.getcwd()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
influence_path = os.path.join(project_root, 'utilities', 'influence')
if influence_path not in sys.path:
    sys.path.append(influence_path)

def s_test(z_test, t_test, model, z_loader, device=-1, damp=0.01, scale=25.0,
           recursion_depth=5000, eps=2e-4, patience=None):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product. Added early stopping

    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        device: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test"""
    #print(f"z_test: {z_test}, t_test: {t_test}")
    v = grad_z(z_test, t_test, model, device)
    h_estimate = [t.clone().detach() for t in v]
    stoppingcounter=0
    for i in range(recursion_depth):
        # Get a new random minibatch each step
        print(f"Recursion step {i} in s_test")
        try:
            x, t = next(z_loader_iter)
        except NameError:
            z_loader_iter = iter(z_loader)
            x, t = next(z_loader_iter)
        except StopIteration:
            z_loader_iter = iter(z_loader)
            x, t = next(z_loader_iter)
        if device >= 0:
            x, t = x.cuda(), t.cuda()
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"x has nans/infs")
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"t has nans/infs")
        y = model(x)
        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"y has nans/infs")
        loss = calc_loss(y, t)
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"loss has nans/infs")
        params = [p for p in model.parameters() if p.requires_grad]
        if any(torch.isnan(p).any() or torch.isinf(p).any() for p in params):
            print("Some parameter contains NaNs/infs")
        hv = hvp(loss, params, h_estimate)
        if any(torch.isnan(h).any() or torch.isinf(h).any() for h in hv):
            print("Some of hv contains NaNs/infs")
            #print(hv)

        # Update h_estimate without tracking gradients
        with torch.no_grad():
            h_new = [_v + (1 - damp) * _h_e - _hv / scale
                     for _v, _h_e, _hv in zip(v, h_estimate, hv)]

        # Optional: early stopping if h_estimate converges
        delta = sum((_h_e - _h_n).norm() / (_h_e.norm() + 1e-9) for _h_e, _h_n in zip(h_estimate, h_new))
        h_estimate = h_new

        if any(torch.isnan(h).any() for h in h_estimate):
            print("Some of h_estimate contains NaNs")
        if patience is not None:
            if delta<eps:
                stoppingcounter+=1
                if stoppingcounter >= patience:
                    #print(f"Early stopping at recursion {i} after {patience} stable steps.")
                    break
            else:
                stoppingcounter=0
        #display_progress("Calc. s_test recursions: ", i, recursion_depth, h_estimate)
    return h_estimate

def calc_loss(y, t):
    """Calculates the loss

    Arguments:
        y: torch tensor, input with size (minibatch, nr_of_classes)
        t: torch tensor, target expected by loss of size (0 to nr_of_classes-1)

    Returns:
        loss: scalar, the loss"""
    ####################
    # if dim == [0, 1, 3] then dim=0; else dim=1
    ####################
    # y = torch.nn.functional.log_softmax(y, dim=0)
    # For classification: y shape is (batch_size, num_classes), so use dim=1
    y = torch.nn.functional.log_softmax(y, dim=1)
    loss = torch.nn.functional.nll_loss(
        y, t, weight=None, reduction='mean')
    return loss


def grad_z(z, t, model, device=-1):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        t: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        device: int, device id to use for GPU, -1 for CPU

    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    model.eval()
    # initialize
    if device >= 0:
        z, t = z.cuda(), t.cuda()
    y = model(z)
    loss = calc_loss(y, t)
    # Compute sum of gradients from model parameters to loss
    params = [ p for p in model.parameters() if p.requires_grad ]
    grads = grad(loss, params, create_graph=True, allow_unused=True)
    # Replace None gradients (unused parameters) with zero tensors
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
    return list(grads)


def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))
    logging.info("hvp computation")
    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True, allow_unused=True)
    # Replace None gradients with zero tensors
    first_grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(first_grads, w)]
    if any(torch.isnan(f).any() or torch.isinf(f).any() for f in first_grads):
        print("Some of first_grads contains NaNs/infs")
    #print(f"first_grads: {first_grads}")
    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)
    if torch.isnan(elemwise_products).any() or torch.isinf(elemwise_products).any():
        print(f"elemwise_products has nans/infs")
        print(elemwise_products)
    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True, allow_unused=True)
    return_grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(return_grads, w)]
    if any(torch.isnan(g).any() or torch.isinf(g).any() for g in return_grads):
        print("Some of return_grads contains NaNs/infs")
        print(return_grads)
    return return_grads