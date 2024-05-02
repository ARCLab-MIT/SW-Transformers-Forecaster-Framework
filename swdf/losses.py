# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/losses.ipynb.

# %% auto 0
__all__ = ['Loss', 'MSELoss', 'MAELoss', 'MSLELoss', 'HubberLoss', 'WeightedLoss', 'wMSELoss', 'wMAELoss', 'wMSLELoss',
           'wHubberLoss', 'ClassificationLoss', 'TrendedLoss', 'LossMetrics']

# %% ../nbs/losses.ipynb 0
from abc import ABC, abstractmethod
import torch
import numpy as np
import pandas as pd
from tsai.basics import *

# %% ../nbs/losses.ipynb 3
class Loss(nn.Module, ABC):
    def __init__(self, reduction:str=None):
        super().__init__()
        self.reduction = reduction
    
    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum': return loss.sum()
        return loss
    
    @abstractmethod
    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return NotImplementedError
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self._compute_loss(input, target)
        return self._reduce(loss)

# %% ../nbs/losses.ipynb 4
class MSELoss(Loss):
    def __init__(self, reduction:str=None):
        super().__init__(reduction)

    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (target-input)**2

class MAELoss(Loss):
    def __init__(self, reduce:str=None):
        super().__init__()
        self.reduce = reduce

    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.abs(target-input)
    
class MSLELoss(Loss):
    def __init__(self, reduction:str=None):
        super().__init__(reduction)

    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (torch.log1p(target) - torch.log1p(input))**2
    
class HubberLoss(Loss):
    def __init__(self, reduction:str=None, delta:float=1.):
        super().__init__(reduction)
        self.delta = delta

    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = target - input
        
        is_small_error = error < self.delta
        small_error_loss = (0.5 * (error ** 2))
        large_error_loss = (self.delta * (torch.abs(error) - 0.5 * self.delta))

        return torch.where(is_small_error, small_error_loss, large_error_loss)

# %% ../nbs/losses.ipynb 6
class WeightedLoss(nn.Module, ABC):
    def __init__(self, ranges:ndarray, weights:ndarray):
        super().__init__()
        self.register_buffer('ranges', torch.Tensor(ranges))
        self.register_buffer('weights', torch.Tensor(weights))

    def weighted_loss_tensor(self, target: torch.Tensor) -> torch.Tensor:        
        batch, variables, horizon = target.shape  # Example shape (32, 4, 6)
        variable, range, interval = self.ranges.shape  # Example shape (4, 4, 2)

        target_shaped = torch.reshape(target, (batch, variables, 1, horizon))  # Example shape (32, 4, 6) -> (32, 4, 1, 6)
        ranges_shaped = torch.reshape(self.ranges, (variable, range, 1, interval))  # Example shape (4, 4, 2) -> (4, 4, 1, 2)

        weights_tensor = ((ranges_shaped[..., 0] <= target_shaped) & (target_shaped <= ranges_shaped[..., 1])).float()
        
        return torch.einsum('r,bvrh->bvh', self.weights, weights_tensor)
    
    @abstractmethod
    def loss_measure(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return NotImplementedError
    
    def forward(self, y_pred, y_true):
        error = self.loss_measure(y_pred, y_true)
        weights = self.weighted_loss_tensor(y_true)
        loss = (error * weights).mean()
        
        return loss

# %% ../nbs/losses.ipynb 8
class wMSELoss(WeightedLoss):
    def __init__(self, ranges, weights):
        super().__init__(ranges, weights)

    
    def loss_measure(self, input, target):
        return MSELoss()(input, target)

# %% ../nbs/losses.ipynb 9
class wMAELoss(WeightedLoss):
    def __init__(self, ranges, weights):
        super().__init__(ranges, weights)

    def loss_measure(self, input, target):
        return MAELoss()(input, target)

# %% ../nbs/losses.ipynb 10
class wMSLELoss(WeightedLoss):
    def __init__(self, ranges, weights):
        super().__init__(ranges, weights)
    
    def loss_measure(self, input, target):
        return MSLELoss()(input, target)

# %% ../nbs/losses.ipynb 11
class wHubberLoss(WeightedLoss):
    def __init__(self, ranges, weights, delta=2.0):
        super().__init__(ranges, weights)
        self.delta = delta
    
    def loss_measure(self, y_pred, y_true):
        return HubberLoss(self.delta)(y_pred, y_true)

# %% ../nbs/losses.ipynb 12
class ClassificationLoss(WeightedLoss):
    def __init__(self, ranges, loss):
        n_variables = ranges.shape[1]
        weights = np.arange(n_variables)

        super().__init__(ranges, weights)

        self.loss = loss
    
    def loss_measure(self, input, target):
        return self.loss(input, target)

    def forward(self, input, target):
        error = self.loss_measure(input, target)
        weights = 1 + torch.abs(self.weighted_loss_tensor(target) - self.weighted_loss_tensor(input))
        loss = (error * weights).mean()
        
        return loss

# %% ../nbs/losses.ipynb 13
class TrendedLoss(nn.Module):
    def __init__(self, loss: Loss):
        super().__init__()
        self.loss = loss

    @staticmethod
    def _slope(y):
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, deg=1)
        return slope

    @staticmethod
    def _calculate_trends(tensor):
        np_tensor = tensor.cpu().detach().numpy()
        trends = np.apply_along_axis(TrendedLoss._slope, 2, np_tensor)
        return torch.Tensor(trends)

    def forward(self, input, target):
        batch, variables, _ = input.shape

        input_trend = TrendedLoss._calculate_trends(input)
        target_trend = TrendedLoss._calculate_trends(target)
        
        trend_diff = 1 + torch.abs(input_trend - target_trend)

        error = self.loss(input, target)
        weights = trend_diff.reshape(batch,variables,1)
        loss = (error * weights).mean()

        return loss

# %% ../nbs/losses.ipynb 16
class LossMetrics:
    def __init__(self, loss_func:WeightedLoss, solact_levels:list):
        self.loss_func = loss_func
        self.solact_levels = solact_levels

    # Weighted Regressive Loss Metrics
    def _apply_weighted_loss_by_level(self, input, target, weight_idx):
        loss_copy = deepcopy(self.loss_func)

        for idx in range(len(loss_copy.weights)):
            if idx != weight_idx:
                loss_copy.weights[idx] = 0
        
        return loss_copy(input, target)
    
    
    # Classification Loss Metrics
    def _compute_misclassifications(self, predictions, targets):
        classifier = self.loss_func.weighted_loss_tensor
        true_labels = classifier(targets)
        predicted_labels = classifier(predictions)

        misclassified_labels = (true_labels != predicted_labels).int() * predicted_labels

        return misclassified_labels.unique(return_counts=True)

    def _count_misclassifications_by_level(self, predictions, targets, level):
        unique_labels, label_counts = self._compute_misclassifications(predictions, targets)
        label_count_dict = dict(zip(unique_labels.tolist(), label_counts.tolist()))

        return label_count_dict.get(level, 0)
    
    

    # Metrics generation
    def _generate_loss_functions(self, loss_func, offset=0):
        metrics = []
        for i, level in enumerate(self.solact_levels):
            def loss_fn(self, input, target, i=i):
                return loss_func(input, target, i+offset)

            method_name = f"loss_{level}"
            loss_fn.__name__ = method_name
            setattr(self, method_name, types.MethodType(loss_fn, self))
            metrics.append(getattr(self, method_name))
        return metrics

    def get_metrics(self):
        if isinstance(self.loss_func, ClassificationLoss):
            return self._generate_loss_functions(self._count_misclassifications_by_level, offset=1)
        
        elif isinstance(self.loss_func, (TrendedLoss, Loss)):
            return []
        
        else:
            return self._generate_loss_functions(self._apply_weighted_loss_by_level)

