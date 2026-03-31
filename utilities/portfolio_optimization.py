"""
Portfolio optimization utilities.

Implements:
- Minimum variance portfolio (closed-form solution)
- Mean-variance portfolio (closed-form solution)

References:
- Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance.
"""

import numpy as np
from utilities.covariance_utilities import (
    _validate_covariance_matrix,
)


def minimum_variance_portfolio(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the minimum variance portfolio weights given a covariance matrix.
    In particular the weights are given by:
    w = (Σ * 1) / (1^T * Σ * 1), i.e. the solution of the optimization problem:
    min_w w^T * Σ * w, subject to 1^T * w = 1.

    Parameters:
        cov_matrix (np.ndarray): Covariance matrix of asset returns.

    Returns:
        np.ndarray: Weights of the minimum variance portfolio.
    """
    cov_matrix = _validate_covariance_matrix(
        cov_matrix,
        name="cov_matrix",
        require_positive_definite=True,
        positive_definite_message=(
            "cov_matrix must be positive definite (symmetric with positive eigenvalues)"
        ),
    )

    n = cov_matrix.shape[0]
    ones_vec = np.ones((n, 1))

    inv_cov = np.linalg.inv(cov_matrix)
    min_var_ptf_numerator = inv_cov @ ones_vec
    min_var_ptf_weights = min_var_ptf_numerator / (ones_vec.T @ min_var_ptf_numerator)

    return min_var_ptf_weights.flatten()


def mean_variance_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 1.0,
) -> np.ndarray:
    """
    Calculate the classic mean-variance portfolio weights given expected returns and a
    covariance matrix.

    In particular the weights solve:
    max_w mu^T * w - (gamma / 2) * w^T * Sigma * w, subject to 1^T * w = 1,
    where mu are the expected returns and gamma is the risk-aversion parameter.

    Parameters:
        expected_returns (np.ndarray): Expected returns vector.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
        risk_aversion (float): Risk-aversion parameter gamma. Must be strictly positive.

    Returns:
        np.ndarray: Weights of the mean-variance portfolio.
    """
    cov_matrix = _validate_covariance_matrix(
        cov_matrix,
        name="cov_matrix",
        require_positive_definite=True,
        positive_definite_message=(
            "cov_matrix must be positive definite (symmetric with positive eigenvalues)"
        ),
    )

    expected_returns = np.asarray(expected_returns, dtype=float)
    if expected_returns.ndim == 2 and 1 in expected_returns.shape:
        expected_returns = expected_returns.reshape(-1)
    elif expected_returns.ndim != 1:
        raise ValueError(
            "expected_returns must be one-dimensional or a single-column vector"
        )

    if expected_returns.shape[0] != cov_matrix.shape[0]:
        raise ValueError(
            "expected_returns and cov_matrix must refer to the same number of assets, "
            f"got {expected_returns.shape[0]} and {cov_matrix.shape[0]}"
        )

    if not np.isfinite(expected_returns).all():
        raise ValueError("expected_returns contains NaN or Inf values")

    if not np.isfinite(risk_aversion):
        raise ValueError("risk_aversion must be finite")

    if risk_aversion <= 0:
        raise ValueError(
            f"risk_aversion must be strictly positive, got {risk_aversion}"
        )

    n = cov_matrix.shape[0]
    ones_vec = np.ones(n)
    inv_cov = np.linalg.inv(cov_matrix)

    # Unconstrained MV portfolio: (1/γ) * Σ⁻¹ * μ
    mv_unconstrained = (1.0 / risk_aversion) * inv_cov @ expected_returns

    # Min variance portfolio (for the constraint)
    inv_cov_ones = inv_cov @ ones_vec
    w_min_var = inv_cov_ones / (ones_vec @ inv_cov_ones)

    # Adjust to satisfy sum-to-one constraint
    mean_var_ptf_weights = w_min_var + mv_unconstrained - (ones_vec @ mv_unconstrained) * w_min_var

    return mean_var_ptf_weights.flatten()
