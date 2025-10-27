"""Robust loss functions for outlier handling in ICP."""

import numpy as np


def huber_loss_weights(residuals, delta=1.0):
    """
    Compute Huber loss weights for robust estimation.
    
    Good for handling 10-20% outliers. Transitions from quadratic to linear
    penalty at the delta threshold.
    
    Args:
        residuals: Array of distances/errors
        delta: Threshold for switching from quadratic to linear
    
    Returns:
        Array of weights (0-1) for each correspondence
    """
    weights = np.ones_like(residuals)
    outlier_mask = residuals > delta
    weights[outlier_mask] = delta / residuals[outlier_mask]
    return weights


def tukey_loss_weights(residuals, c=4.685):
    """
    Compute Tukey biweight loss weights for robust estimation.
    
    Very robust to severe outliers (handles 30-50% outliers). Completely
    rejects correspondences beyond threshold c.
    
    Args:
        residuals: Array of distances/errors
        c: Tuning constant (4.685 for 95% efficiency)
    
    Returns:
        Array of weights (0-1) for each correspondence
    """
    normalized = residuals / c
    weights = np.zeros_like(residuals)
    inlier_mask = normalized <= 1.0
    weights[inlier_mask] = (1 - normalized[inlier_mask]**2)**2
    return weights


def percentile_filter_weights(distances, percentile=90):
    """
    Filter correspondences by distance percentile.
    
    Simple outlier rejection: keeps only the best N% of correspondences.
    Fast and effective for many scenarios.
    
    Args:
        distances: Array of correspondence distances
        percentile: Keep only correspondences below this percentile
    
    Returns:
        Binary weights (0 or 1)
    """
    threshold = np.percentile(distances, percentile)
    return (distances <= threshold).astype(float)


def get_loss_function(loss_fn='none', loss_params=None):
    """
    Get loss function and parameters.
    
    Args:
        loss_fn: One of 'none', 'huber', 'tukey', 'percentile'
        loss_params: Dictionary of loss-specific parameters
    
    Returns:
        Tuple of (loss_function, params_dict)
    """
    if loss_params is None:
        loss_params = {}
    
    loss_functions = {
        'none': (None, {}),
        'huber': (huber_loss_weights, {'delta': loss_params.get('delta', 10.0)}),
        'tukey': (tukey_loss_weights, {'c': loss_params.get('c', 15.0)}),
        'percentile': (percentile_filter_weights, {'percentile': loss_params.get('percentile', 85)})
    }
    
    return loss_functions.get(loss_fn, (None, {}))

