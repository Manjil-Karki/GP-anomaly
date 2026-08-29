"""GPyTorch ExactGP with 6-kernel sweep and L-BFGS optimisation.

Six kernels compared per fold via LML; best retained.
torch/gpytorch imported lazily so phases 0-1 run without the GPU stack.

Kernel roster:
  rbf      — Squared Exponential (ARD): smooth, infinitely differentiable
  mat12    — Matérn-1/2 (ARD): rough, once differentiable
  mat32    — Matérn-3/2 (ARD): moderately smooth
  mat52    — Matérn-5/2 (ARD): smooth enough for twice-differentiable targets
  rq       — Rational Quadratic (ARD): mixture of length-scales
  lin_rbf  — Linear × RBF composite: captures global trends + local variation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import GP_LR, GP_MAX_ITER, GP_N_RESTARTS, RANDOM_SEED

log = logging.getLogger(__name__)

KERNEL_NAMES = ["rbf", "mat12", "mat32", "mat52", "rq", "lin_rbf"]


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------

def _make_covar(kernel_name: str, d: int):
    from gpytorch.kernels import (
        RBFKernel, MaternKernel, RQKernel, LinearKernel, ScaleKernel,
    )
    if kernel_name == "rbf":
        return ScaleKernel(RBFKernel(ard_num_dims=d))
    if kernel_name == "mat12":
        return ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=d))
    if kernel_name == "mat32":
        return ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=d))
    if kernel_name == "mat52":
        return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))
    if kernel_name == "rq":
        return ScaleKernel(RQKernel(ard_num_dims=d))
    if kernel_name == "lin_rbf":
        return ScaleKernel(LinearKernel()) + ScaleKernel(RBFKernel(ard_num_dims=d))
    raise ValueError(f"Unknown kernel: {kernel_name}")


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def _build_gp_class():
    import torch
    import gpytorch
    from gpytorch.distributions import MultivariateNormal
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.means import ConstantMean
    from gpytorch.models import ExactGP

    class SeverityGP(ExactGP):
        def __init__(self, train_x, train_y, likelihood, covar_module):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module  = ConstantMean()
            self.covar_module = covar_module

        def forward(self, x):
            return MultivariateNormal(self.mean_module(x), self.covar_module(x))

    return SeverityGP, GaussianLikelihood


# ---------------------------------------------------------------------------
# Fitted GP container
# ---------------------------------------------------------------------------

@dataclass
class FittedGP:
    model:       object   # SeverityGP
    likelihood:  object   # GaussianLikelihood
    device:      str
    lml:         float
    kernel_name: str = "rbf"


# ---------------------------------------------------------------------------
# Single-kernel fitting
# ---------------------------------------------------------------------------

def _fit_single_kernel(
    train_x,        # torch.Tensor (N, d) float64 on device
    train_y,        # torch.Tensor (N,) float64 on device
    kernel_name: str,
    device: str,
    rng: np.random.Generator,
    n_restarts: int,
    max_iter: int,
):
    """
    Fit one kernel with n_restarts L-BFGS initialisations.
    Returns (model, likelihood, best_lml).
    """
    import torch
    from gpytorch.mlls import ExactMarginalLogLikelihood

    SeverityGP, GaussianLikelihood = _build_gp_class()
    d = train_x.shape[1]

    best_lml: float = -np.inf
    best_state: Optional[dict] = None

    for restart in range(n_restarts):
        covar = _make_covar(kernel_name, d).to(device).double()
        lik   = GaussianLikelihood().to(device).double()
        model = SeverityGP(train_x, train_y, lik, covar).to(device).double()

        if restart > 0:
            with torch.no_grad():
                lik.noise = torch.tensor(float(rng.uniform(1e-4, 0.5)), dtype=torch.float64, device=device)
                # Randomise outputscale
                for mod in model.covar_module.modules():
                    if hasattr(mod, "outputscale"):
                        mod.outputscale = torch.tensor(float(rng.uniform(0.1, 5.0)), dtype=torch.float64, device=device)
                    if hasattr(mod, "lengthscale") and mod.lengthscale is not None:
                        val = rng.uniform(0.1, 5.0, size=mod.lengthscale.shape)
                        mod.lengthscale = torch.tensor(val, dtype=torch.float64, device=device)

        model.train(); lik.train()
        mll = ExactMarginalLogLikelihood(lik, model)
        opt = torch.optim.LBFGS(
            list(model.parameters()) + list(lik.parameters()),
            lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe",
        )

        final_loss = torch.tensor(float("inf"))

        def closure():
            nonlocal final_loss
            opt.zero_grad()
            try:
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                final_loss = loss
            except Exception:
                final_loss = torch.tensor(float("inf"))
            return final_loss

        try:
            opt.step(closure)
        except Exception as e:
            log.debug(f"  {kernel_name} restart {restart}: LBFGS failed ({e})")
            continue

        lml = -float(final_loss.item())
        if lml > best_lml:
            best_lml = lml
            best_state = {
                "model": {k: v.clone() for k, v in model.state_dict().items()},
                "lik":   {k: v.clone() for k, v in lik.state_dict().items()},
                "covar": covar,
            }
        log.debug(f"  {kernel_name} restart {restart+1}/{n_restarts}  LML={lml:.4f}")

    if best_state is None:
        return None, None, -np.inf

    covar = best_state["covar"].to(device).double()
    best_lik   = GaussianLikelihood().to(device).double()
    best_model = SeverityGP(train_x, train_y, best_lik, covar).to(device).double()
    best_model.load_state_dict(best_state["model"])
    best_lik.load_state_dict(best_state["lik"])
    best_model.eval(); best_lik.eval()
    return best_model, best_lik, best_lml


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: Optional[str] = None,
    n_restarts: int = GP_N_RESTARTS,
    max_iter: int = GP_MAX_ITER,
    kernels: list[str] = KERNEL_NAMES,
) -> FittedGP:
    """
    Fit ExactGP with L-BFGS across n_restarts × len(kernels) initialisations.
    Kernel selection by marginal likelihood maximisation.

    X_train : (N, d) float32  — PCA-reduced embeddings
    y_train : (N,)   float64  — log(severity)
    Returns FittedGP with best kernel and highest LML.
    """
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_x = torch.tensor(X_train, dtype=torch.float64).to(device)
    train_y = torch.tensor(y_train, dtype=torch.float64).to(device)

    rng = np.random.default_rng(RANDOM_SEED)
    best_lml = -np.inf
    best_fitted: Optional[FittedGP] = None

    for kname in kernels:
        model, lik, lml = _fit_single_kernel(
            train_x, train_y, kname, device, rng, n_restarts, max_iter
        )
        log.info(f"  kernel={kname}  LML={lml:.4f}")
        if model is not None and lml > best_lml:
            best_lml = lml
            best_fitted = FittedGP(
                model=model, likelihood=lik,
                device=device, lml=lml, kernel_name=kname,
            )

    if best_fitted is None:
        raise RuntimeError("All kernels failed to fit. Check data and device.")

    log.info(f"  Best kernel: {best_fitted.kernel_name}  LML={best_fitted.lml:.4f}")
    return best_fitted


def predict_gp(
    X_test: np.ndarray,
    fitted: FittedGP,
) -> tuple[np.ndarray, np.ndarray]:
    """
    GP posterior predictive mean and variance at test points.
    Returns (mean, variance), each (N,).
    High variance → out-of-distribution (novel defect type).
    """
    import torch
    import gpytorch

    test_x = torch.tensor(X_test, dtype=torch.float64).to(fitted.device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = fitted.likelihood(fitted.model(test_x))
    return (
        pred.mean.cpu().numpy().astype(np.float64),
        pred.variance.cpu().numpy().astype(np.float64),
    )
