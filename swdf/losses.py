# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/losses.ipynb.

# %% auto 0
__all__ = ['Loss', 'MSELoss', 'MAELoss', 'MSLELoss', 'RMSLELoss', 'HubberLoss', 'QuantileLoss', 'WeightedLoss', 'wMSELoss',
           'wMAELoss', 'wMSLELoss', 'wRMSLELoss', 'wHubberLoss', 'wQuantileLoss', 'ClassificationLoss', 'TrendedLoss',
           'LossFactory']

# %% ../nbs/losses.ipynb 0
from abc import ABC, abstractmethod
import torch
import numpy as np
import pandas as pd
from tsai.basics import *
from IPython.display import HTML, display

# %% ../nbs/losses.ipynb 3
class Loss(nn.Module, ABC):
    """
    <p>Base class for loss functions, providing a common interface for different types of losses.</p>
    <h3>Attributes:</h3>
    <ul>
        <li>reduction (str): Method for reducing the loss value across batches | <i><u>Default</u></i>: None.</li>
    </ul>
    """
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
    """
    <p>Mean Squared Error Loss (MSELoss) measures the average squared difference between predicted and actual values.</p>
    <h3>Attributes:</h3>
    <ul>
        <li>reduction (str): Method for reducing the loss value across batches | <i><u>Default</u></i>: None.</li>
    </ul>
    """

    def __init__(self, reduction:str=None):
        super().__init__(reduction)

    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (target-input)**2

# %% ../nbs/losses.ipynb 8
class MAELoss(Loss):
    """
    <p>Mean Absolute Error Loss (MAELoss) calculates the average absolute differences between predicted and actual values.</p>
    <h3>Attributes:</h3>
    <ul>
        <li>reduction (str): Method for reducing the loss value across batches | <i><u>Default</u></i>: None.</li>
    </ul>
    """
    def __init__(self, reduction:str=None):
        super().__init__(reduction)

    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.abs(target-input)
    

# %% ../nbs/losses.ipynb 10
class MSLELoss(Loss):
    """
    <p>Mean Squared Logarithmic Error Loss (MSLELoss) penalizes underestimations more than overestimations by using logarithms.</p>
    <h3>Attributes:</h3>
    <ul>
        <li>reduction (str): Method for reducing the loss value across batches | <i><u>Default</u></i>: None.</li>
    </ul>
    """
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
        target_scaled = MSLELoss.inverse_scale_values_below_threshold(target, -1, -1 + epsilon, -0.1)
        input_scaled = MSLELoss.inverse_scale_values_below_threshold(input, -1, -1 + epsilon, -0.1)

        return (torch.log1p(target) - torch.log1p(input))**2

# %% ../nbs/losses.ipynb 12
class RMSLELoss(nn.Module):
    """
    <p>Root Mean Squared Logarithmic Error Loss (RMSLELoss) is the square root of MSLE, useful for reducing the impact of outliers.</p>
    <h3>Attributes:</h3>
    <ul>
        <li>reduction (str): Method for reducing the loss value across batches | <i><u>Default</u></i>: None.</li>
    </ul>
    """

    def __init__(self, reduction:str='mean'):
        super().__init__()
        self.msle_loss = MSLELoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor, reduction='mean') -> torch.Tensor:
        return torch.sqrt(self.msle_loss(input, target, reduction))

# %% ../nbs/losses.ipynb 14
class HubberLoss(Loss):
    """
    <p>Huber Loss (HL) combines the characteristics of both MSE and MAE, aiming to benefit from their respective strengths while mitigating their limitations.</p>
    <h3>Attributes:</h3>
    <ul>
        <li>reduction (str): Method for reducing the loss value across batches | <i><u>Default</u></i>: 'mean'.</li>
        <li>delta (float): Threshold from where the loss changes from MAE to MSE-like functioning.</li>
    </ul>
    """
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
    """
    <p>Quantile Loss is used for regression tasks where we want to predict a specific quantile.</p>
    <h3>Attributes:</h3>
    <ul>
        <li>reduction (str): Method for reducing the loss value across batches | <i><u>Default</u></i>: None.</li>
        <li>quantile (float): The quantile to be predicted, usually a value between 0 and 1.</li>
    </ul>
    """
    def __init__(self, quantile: float, reduction: str = None):
        super().__init__(reduction)
        self.quantile = quantile

    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        errors = target - input
        return torch.where(errors >= 0, self.quantile * errors, (self.quantile - 1) * errors)

# %% ../nbs/losses.ipynb 18
class WeightedLoss(nn.Module, ABC):
    """
    <p>Base class for weighted loss functions, where different samples are given different importance.</p>
    <h3>Attributes:</h3>
    <ul>
        <li>reduction (str): Method for reducing the loss value across batches | <i><u>Default</u></i>: None.</li>
        <li>weights (Tensor): Weights assigned to each sample in the batch.</li>
        <li>thresholds (Tensor): Threshold values for weighted computation.</li>
    </ul>
    """
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
        if (self.all_variables_have_same_weights):
            ranges = np.array(list(thresholds.values())[:])
            weights = np.array(next(iter(weights.values())))
        
        # If each variable has its own weights, calculate the maximum size of weights.
        # Padding shorter weights with NaNs prevents heterogeneous tensor errors.
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
    """
    <p>Weighted Mean Squared Error Loss (wMSELoss) is the weighted version of MSE, giving different importance to different samples.</p>
    <h3>Attributes:</h3>
    <ul>
        <li>reduction (str): Method for reducing the loss value across batches | <i><u>Default</u></i>: None.</li>
        <li>weights (Tensor): Weights assigned to each sample in the batch.</li>
        <li>thresholds (Tensor): Threshold values for weighted computation.</li>
    </ul>
    """
    def __init__(self, thresholds, weights):
        super().__init__(thresholds, weights)

    
    def loss_measure(self, input, target):
        return MSELoss()(input, target)
    

    
class wMAELoss(WeightedLoss):
    """
    <p>Weighted Mean Absolute Error Loss (wMAELoss) is the weighted version of MAE, giving different importance to different samples.</p>
    <h3>Attributes:</h3>
    <ul>
        <li>reduction (str): Method for reducing the loss value across batches | <i><u>Default</u></i>: None.</li>
        <li>weights (Tensor): Weights assigned to each sample in the batch.</li>
        <li>thresholds (Tensor): Threshold values for weighted computation.</li>
    </ul>
    """
    def __init__(self, thresholds, weights):
        super().__init__(thresholds, weights)

    def loss_measure(self, input, target):
        return MAELoss()(input, target)


    
class wMSLELoss(WeightedLoss):
    """
    <p>Weighted Mean Squared Logarithmic Error Loss (wMSLELoss) is the weighted version of MSLE, penalizing underestimations more than overestimations.</p>
    <h3>Attributes:</h3>
    <ul>
        <li>reduction (str): Method for reducing the loss value across batches | <i><u>Default</u></i>: None.</li>
        <li>weights (Tensor): Weights assigned to each sample in the batch.</li>
        <li>thresholds (Tensor): Threshold values for weighted computation.</li>
    </ul>
    """
    def __init__(self, thresholds, weights):
        super().__init__(thresholds, weights)
    
    def loss_measure(self, input, target):
        return MSLELoss()(input, target)
    


class wRMSLELoss(nn.Module):
    """
    <p>Weighted Root Mean Squared Logarithmic Error Loss (wRMSLELoss) is the weighted version of RMSLE, useful for reducing the impact of outliers.</p>
    <h3>Attributes:</h3>
    <ul>
        <li>reduction (str): Method for reducing the loss value across batches | <i><u>Default</u></i>: None.</li>
        <li>weights (Tensor): Weights assigned to each sample in the batch.</li>
        <li>thresholds (Tensor): Threshold values for weighted computation.</li>
    </ul>
    """
    def __init__(self, thresholds, weights):
        super().__init__()
        self.msle_loss = wMSLELoss(thresholds, weights)
        
    def forward(self, input, target, reduction='mean'):
        return torch.sqrt(self.msle_loss(input, target, reduction))
    


class wHubberLoss(WeightedLoss):
    """
    <p>Weighted Huber Loss (wHubberLoss) combines the characteristics of both MSE and MAE, with weights for different samples.</p>
    <h3>Attributes:</h3>
    <ul>
        <li>reduction (str): Method for reducing the loss value across batches | <i><u>Default</u></i>: 'mean'.</li>
        <li>delta (float): Threshold from where the loss changes from MAE to MSE-like functioning.</li>
        <li>weights (Tensor): Weights assigned to each sample in the batch.</li>
        <li>thresholds (Tensor): Threshold values for weighted computation.</li>
    </ul>
    """
    def __init__(self, thresholds, weights, delta=2.0):
        super().__init__(thresholds, weights)
        self.delta = delta
    
    def loss_measure(self, y_pred, y_true):
        return HubberLoss(delta=self.delta)(y_pred, y_true)
    


class wQuantileLoss(WeightedLoss):
    """
    <p>Weighted Quantile Loss is used for regression tasks with weighted samples where we want to predict a specific quantile.</p>
    <h3>Attributes:</h3>
    <ul>
        <li>reduction (str): Method for reducing the loss value across batches | <i><u>Default</u></i>: None.</li>
        <li>quantile (float): The quantile to be predicted, usually a value between 0 and 1.</li>
        <li>weights (Tensor): Weights assigned to each sample in the batch.</li>
        <li>thresholds (Tensor): Threshold values for weighted computation.</li>
    </ul>
    """
    def __init__(self, thresholds, weights, quantile=0.5):
        super().__init__(thresholds, weights)
        self.quantile = quantile
    
    def loss_measure(self, y_pred, y_true):
        return QuantileLoss(quantile=self.quantile)(y_pred, y_true)

# %% ../nbs/losses.ipynb 23
class ClassificationLoss(WeightedLoss):
    """
    <p>Loss function for classification tasks, suitable for handling imbalanced classes and other classification-specific challenges.</p>
    <h3>Attributes:</h3>
    <ul>
        <li>reduction (str): Method for reducing the loss value across batches | <i><u>Default</u></i>: None.</li>
        <li>primary_loss (Loss): The base loss function used for classification.</li>
        <li>alpha (float): Weighting factor for balancing the importance of different classes.</li>
    </ul>
    """
    def __init__(self, thresholds, primary_loss, multivar_weights=False, alpha=0.5):
        if multivar_weights:
            weights = {key: np.arange(len(thresholds[key])) for key in thresholds.keys()}
        else:
            first_values = next(iter(thresholds.values()), None) 
            weights = {'All': np.arange(len(first_values))}
        super().__init__(thresholds, weights)

        self.loss = primary_loss

        if alpha < 0 or alpha > 1:
            raise ValueError('Alpha must be between 0 and 1, as it is the weight of the categorical loss against the other loss.')
        self.alpha = alpha
    
    def loss_measure(self, input, target):
        primary_loss_value = self.loss(input, target, reduction=None)

        categorical_error = torch.abs(self.weighted_loss_tensor(target) - self.weighted_loss_tensor(input))
        categorical_loss_value = torch.mean(categorical_error, dim=2, keepdim=True)


        return (1 - self.alpha) * primary_loss_value + self.alpha * categorical_loss_value


    def forward(self, input, target, reduction='mean'):
        error = self.loss_measure(input, target)

        # if (error.shape != weights.shape): # To properly format the weights tensor in case of multi-variable classification
          #   weights = weights.mean(dim=1)
            
        if reduction == 'mean':
            loss = error.mean()
        elif reduction == 'sum':
            loss = error.sum()
        else:
            loss = error
        
        return loss

# %% ../nbs/losses.ipynb 25
class TrendedLoss(nn.Module):
    """
    <p>Trended Loss incorporates trends in the data to adjust the loss computation accordingly.</p>
    <h3>Attributes:</h3>
    <ul>
        <li>primary_loss (Loss): The base loss function used in combination with trend adjustments.</li>
    </ul>
    """
    def __init__(self, primary_loss: Loss):
        super().__init__()
        self.loss = primary_loss

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
        weights = trend_diff.reshape(batch,variables,1).to(error.device)
        loss = (error * weights).mean()

        return loss

# %% ../nbs/losses.ipynb 27
class LossFactory:
    losses = {
        'MSE': MSELoss,
        'MAE': MAELoss,
        'MSLE': MSELoss,
        'RMSLE': RMSLELoss,
        'Hubber': HubberLoss,
        'Quantile': QuantileLoss,
        'wMSE': wMSELoss,
        'wMAE': wMAELoss,
        'wMSLE': wMSLELoss,
        'wRMSLE': wRMSLELoss,
        'wHubber': wHubberLoss,
        'wQuantile': wQuantileLoss,
        'Classification': ClassificationLoss,
        'Trended': TrendedLoss
    }

    def __init__(self, thresholds=None, weights=None):
        self.thresholds = thresholds
        self.weights = weights

    @classmethod
    def list(cls):
        table_rows = []
        
        # Generate rows for the table
        for key, value in cls.losses.items():
            doc_html = value.__doc__.strip().replace("\n", " ")
            table_rows.append(f"<tr><td style='text-align: left;'><strong>{key}</strong></td><td style='text-align: left;'>{doc_html}</td></tr>")
        
        # Create the HTML for the table with left-aligned text
        table_html = f"""
        <table>
            <thead>
                <tr>
                    <th style='text-align: left;'>Loss Name</th>
                    <th style='text-align: left;'>Description</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
        """
        
        display(HTML(table_html))



    def create(self, loss_name: str = 'MSE', **kwargs) -> nn.Module:
        """
        <p>Create and return a loss function based on the provided loss name and additional parameters.</p>

        <h3>Parameters:</h3>
        <ul>
            <li><b>loss_name</b> (str): The name of the loss function to create. Default is 'MSE'.</li>
            <li><b>kwargs</b>: Additional keyword arguments specific to certain loss functions.</li>
        </ul>

        <h3>Returns:</h3>
        <p>nn.Module: The instantiated loss function module.</p>

        <h3>Raises:</h3>
        <p>ValueError: If the specified loss function is not found.</p>
        """
        searched_loss = loss_name.lower()
        available_losses = {k.lower(): k for k in LossFactory.losses.keys()}

        if searched_loss not in available_losses:
            raise ValueError(f'Loss {loss_name} not found. Run LossFactory.list() to see available losses.')

        if 'w' in searched_loss and (self.thresholds is not None or self.weights is not None):
            if searched_loss == 'whubber':
                return wHubberLoss(thresholds=self.thresholds, weights=self.weights, delta=kwargs.get('delta', 2.0))
            elif searched_loss == 'wquantile':
                return wQuantileLoss(thresholds=self.thresholds, weights=self.weights, quantile=kwargs.get('quantile', 0.5))
            else:
                return LossFactory.losses[available_losses[searched_loss]](thresholds=self.thresholds, weights=self.weights)

        if searched_loss in ['classification', 'trended']:
            primary_loss = self.create(kwargs.get('primary_loss', 'MSE'), **kwargs)

            if searched_loss == 'classification':
                return ClassificationLoss(
                    thresholds=self.thresholds,
                    primary_loss=primary_loss,
                    multivar_weights=len(self.weights.keys()) > 1,
                    alpha=kwargs.get('alpha', 0.5)
                )
            else:  
                return TrendedLoss(primary_loss=primary_loss)

        if searched_loss == 'hubber':
            return HubberLoss(reduction='mean', delta=kwargs.get('delta', 2.0))

        if searched_loss == 'quantile':
            return QuantileLoss(reduction='mean', quantile=kwargs.get('quantile', 0.5))

        return LossFactory.losses[available_losses[searched_loss]](reduction='mean')
