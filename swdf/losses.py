# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/losses.ipynb.

# %% auto 0
__all__ = ['Loss', 'MSELoss', 'MAELoss', 'MSLELoss', 'RMSLELoss', 'HubberLoss', 'QuantileLoss', 'WeightedLoss', 'wMSELoss',
           'wMAELoss', 'wMSLELoss', 'wRMSLELoss', 'wHubberLoss', 'wQuantileLoss', 'ClassificationLoss', 'TrendedLoss',
           'LossMetrics']

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
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, reduction:str=None) -> torch.Tensor:
        if reduction is not None:
            self.reduction = reduction
        loss = self._compute_loss(input, target)
        return self._reduce(loss)

# %% ../nbs/losses.ipynb 6
class MSELoss(Loss):
    def __init__(self, reduction:str=None):
        super().__init__(reduction)

    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (target-input)**2

# %% ../nbs/losses.ipynb 8
class MAELoss(Loss):
    def __init__(self, reduction:str=None):
        super().__init__(reduction)

    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.abs(target-input)
    

# %% ../nbs/losses.ipynb 10
class MSLELoss(Loss):
    def __init__(self, reduction:str=None):
        super().__init__(reduction)

    @staticmethod
    def inverse_scale_values_below_threshold(tensor, threshold, lower_bound, upper_bound):
        mask = tensor < threshold

        if mask.sum() == 0:
            # If no values are below the threshold, return the original tensor
            return tensor
        
        values_to_scale = tensor[mask]
        min_orig = values_to_scale.min()
        max_orig = values_to_scale.max()
        
        if min_orig == max_orig:
            scaled_values = torch.full_like(tensor, upper_bound)
        else:
            scaled_values = upper_bound - (tensor - min_orig) * (upper_bound - lower_bound
        ) / (max_orig - min_orig)
        
        result_tensor = torch.where(mask, scaled_values, tensor)
        
        return result_tensor

    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        epsilon = torch.finfo(torch.float32).eps
        target_scaled = MSLELoss.inverse_scale_values_below_threshold(target, -1, 0.1, epsilon)
        input_scaled = MSLELoss.inverse_scale_values_below_threshold(input, -1, 0.1, epsilon)
        
        target = torch.where(target <= -1, -1 + target_scaled, target)
        input = torch.where(input <= -1, -1 + input_scaled, input)

        return (torch.log1p(target) - torch.log1p(input))**2

# %% ../nbs/losses.ipynb 12
class RMSLELoss(nn.Module):
    def __init__(self, reduction:str='mean'):
        super().__init__()
        self.msle_loss = MSLELoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor, reduction='mean') -> torch.Tensor:
        return torch.sqrt(self.msle_loss(input, target, reduction))

# %% ../nbs/losses.ipynb 14
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

# %% ../nbs/losses.ipynb 16
class QuantileLoss(Loss):
    def __init__(self, quantile: float, reduction: str = None):
        super().__init__(reduction)
        self.quantile = quantile

    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        errors = target - input
        return torch.where(errors >= 0, self.quantile * errors, (self.quantile - 1) * errors)

# %% ../nbs/losses.ipynb 18
class WeightedLoss(nn.Module, ABC):
    def __init__(self, thresholds:dict, weights:dict):
        super().__init__()

        # Activity levels' weights can be equal across all variables or different,
        # and this should be taken into account during preprocessing. 
        self.all_variables_have_same_weights = len(weights.keys()) == 1
        ranges, weights = self._preprocess_data(thresholds, weights)

        self.register_buffer('ranges', torch.Tensor(ranges))
        self.register_buffer('weights', torch.Tensor(weights))


    def weighted_loss_tensor(self, target: torch.Tensor) -> torch.Tensor:        
        batch, variables, horizon = target.shape  # Example shape (32, 4, 6)
        variable, max_range, interval = self.ranges.shape  # Example shape (4, 4, 2)

        print()

        target_shaped = torch.reshape(target, (batch, variables, 1, horizon))  # Example shape (32, 4, 6) -> (32, 4, 1, 6)
        ranges_shaped = torch.reshape(self.ranges, (variable, max_range, 1, interval))  # Example shape (4, 4, 2) -> (4, 4, 1, 2)

        weights_tensor = ((ranges_shaped[..., 0] <= target_shaped) & (target_shaped <= ranges_shaped[..., 1])).float()
             
        if self.all_variables_have_same_weights:
            equation = 'r,bvrh->bvh'
        else:
            equation = 'vr,bvrh->bvh'

        return torch.einsum(equation, self.weights, weights_tensor)
    
    
    @abstractmethod
    def loss_measure(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return NotImplementedError
    
    def _preprocess_data(self, thresholds, weights):
        # If each variable has its own weights, calculate the maximum size of weights.
        # Padding shorter weights with NaNs prevents heterogeneous tensor errors.
        if (self.all_variables_have_same_weights):
            ranges = np.array(list(thresholds.values())[:])
            weights = np.array(next(iter(weights.values())))
        else:
            def add_padding(x, padding_value, shape):
                result = np.full(shape, padding_value)
                for i, r in enumerate(x):
                    result[i, :len(r)] = r
                return result
            
            max_size = max([len(array) for array in thresholds.values()])

            ranges_raw = thresholds.values()
            ranges = add_padding(ranges_raw, np.nan, (len(ranges_raw), max_size, 2))

            weights_raw = [weights[key] for key in thresholds.keys()]
            weights = add_padding(weights_raw, 0.0, (len(weights_raw), max_size))

        return ranges, weights
    
    
    def forward(self, y_pred, y_true, reduction='mean'):
        error = self.loss_measure(y_pred, y_true)
        weights = self.weighted_loss_tensor(y_true)

        if reduction == 'mean':
            loss = (error * weights).mean()
        elif reduction == 'sum':
            loss = (error * weights).sum()
        else: 
            loss = error*weights
        
        return loss

# %% ../nbs/losses.ipynb 20
class wMSELoss(WeightedLoss):
    def __init__(self, thresholds, weights):
        super().__init__(thresholds, weights)

    
    def loss_measure(self, input, target):
        return MSELoss()(input, target)
    

    
class wMAELoss(WeightedLoss):
    def __init__(self, thresholds, weights):
        super().__init__(thresholds, weights)

    def loss_measure(self, input, target):
        return MAELoss()(input, target)


    
class wMSLELoss(WeightedLoss):
    def __init__(self, thresholds, weights):
        super().__init__(thresholds, weights)
    
    def loss_measure(self, input, target):
        return MSLELoss()(input, target)
    


class wRMSLELoss(nn.Module):
    def __init__(self, thresholds, weights):
        super().__init__()
        self.msle_loss = wMSLELoss(thresholds, weights)
        
    def forward(self, input, target, reduction='mean'):
        return torch.sqrt(self.msle_loss(input, target, reduction))
    


class wHubberLoss(WeightedLoss):
    def __init__(self, thresholds, weights, delta=2.0):
        super().__init__(thresholds, weights)
        self.delta = delta
    
    def loss_measure(self, y_pred, y_true):
        return HubberLoss(delta=self.delta)(y_pred, y_true)
    


class wQuantileLoss(WeightedLoss):
    def __init__(self, thresholds, weights, quantile=0.5):
        super().__init__(thresholds, weights)
        self.quantile = quantile
    
    def loss_measure(self, y_pred, y_true):
        return QuantileLoss(quantile=self.quantile)(y_pred, y_true)

# %% ../nbs/losses.ipynb 23
class ClassificationLoss(WeightedLoss):
    def __init__(self, thresholds, loss):
        n_variables = len(thresholds.keys())
        weights = {'All': np.arange(n_variables)}

        super().__init__(thresholds, weights)

        self.loss = loss
    
    def loss_measure(self, input, target):
        return self.loss(input, target)

    def forward(self, input, target, reduction='mean'):
        error = self.loss_measure(input, target)
        weights = 1 + torch.abs(self.weighted_loss_tensor(target) - self.weighted_loss_tensor(input))

        if (error.shape != weights.shape): # To properly format the weights tensor in case of multi-variable classification
            weights = weights.mean(dim=1)
            
        if reduction == 'mean':
            loss = (error * weights).mean()
        elif reduction == 'sum':
            loss = (error * weights).sum()
        else:
            loss = error * weights
        
        return loss

# %% ../nbs/losses.ipynb 25
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

# %% ../nbs/losses.ipynb 27
class LossMetrics:
    def __init__(self, loss_func, solact_levels):
        self.loss_func = loss_func
        self.solact_levels = solact_levels

    # Weighted Regressive Loss Metrics
    def _apply_weighted_loss_by_level(self, input, target, weight_idx):
        loss_copy = deepcopy(self.loss_func)
        
    
        for idx1 in range(len(loss_copy.weights)):
            if is_iter(loss_copy.weights[0]):
                for idx2 in range(len(loss_copy.weights[idx1])):
                    if (idx1 != weight_idx[0]) | (idx2 != weight_idx[1]):
                        loss_copy.weights[idx1][idx2] = 0
            else:
                if idx1 != weight_idx[1]:
                    loss_copy.weights[idx1] = 0
                
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
    

    # Metrics functions
    ## FSMY Metrics
    def loss_low(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [0,0])
    
    def loss_moderate(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [0,1])
    
    def loss_elevated(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [0,2])
    
    def loss_high(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [0,3])
    
    ## DST-AP Metrics
    def loss_Low(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [0,0])
    
    def loss_Medium(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [0,1])
    
    def loss_Active(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [0,2])
    
    def loss_G0(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [1,0])
    
    def loss_G1(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [1,1])
    
    def loss_G2(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [1,2])
        
    def loss_G3(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [1,3])
    
    def loss_G4(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [1,4])
    
    def loss_G5(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [1,5])
    
    ## ClassificationLoss metrics
    def missclassifications_low(self, predictions, targets):
        return self._count_misclassifications_by_level(predictions, targets, 1)
    
    def missclassifications_moderate(self, predictions, targets):
        return self._count_misclassifications_by_level(predictions, targets, 2)
    
    def missclassifications_elevated(self, predictions, targets):
        return self._count_misclassifications_by_level(predictions, targets, 3)
    
    def missclassifications_high(self, predictions, targets):
        return self._count_misclassifications_by_level(predictions, targets, 4)
    
    ## Metrics Not Available
    def Metrics_Not_Available(self, input, target): return np.nan 
    

    # Metrics retrieval
    def get_metrics(self):
        if isinstance(self.loss_func, ClassificationLoss):
            return [self.missclassifications_low, self.missclassifications_moderate, self.missclassifications_elevated, self.missclassifications_high]
        
        elif isinstance(self.loss_func, WeightedLoss):
            if isinstance(self.solact_levels, list): # FSMY metrics required
                return [self.loss_low, self.loss_moderate, self.loss_elevated, self.loss_high]
            else: # DST-AP metrics required
                return [self.loss_Low, self.loss_Medium, self.loss_Active, self.loss_G0, self.loss_G1, self.loss_G2, self.loss_G3, self.loss_G4, self.loss_G5]
        else:
            return [self.Metrics_Not_Available]

