# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/metrics.ipynb.

# %% auto 0
__all__ = ['Metrics', 'RegressiveMetrics', 'SOLFMYMetrics', 'GEODSTAPMetrics', 'ClassificationMetrics',
           'SOLFMYClassificationMetrics', 'GEODSTAPClassificationMetrics', 'LossMetrics', 'OutlierDetectionMetrics',
           'F1ScoreMetrics', 'AUPRCMetric', 'KSDifferenceMetric', 'AssociationMetrics', 'AccuracyMetrics',
           'BiasMetrics']

# %% ../nbs/metrics.ipynb 0
import sys
sys.path.append('..')
from abc import ABC
import torch
import numpy as np
import pandas as pd
from tsai.basics import *
from .losses import wMAELoss, MSELoss, WeightedLoss, ClassificationLoss
from sklearn.metrics import precision_recall_curve, auc



# %% ../nbs/metrics.ipynb 3
class Metrics(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_metrics(self) -> list:
        return NotImplementedError

# %% ../nbs/metrics.ipynb 6
class RegressiveMetrics(Metrics):
    def __init__(self, loss_func):
        super().__init__()
        self.loss_func = loss_func

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

    @abstractmethod
    def get_metrics(self) -> list:
        return NotImplementedError

# %% ../nbs/metrics.ipynb 8
class SOLFMYMetrics(RegressiveMetrics):
    def __init__(self, loss_func):
        super().__init__(loss_func)
        self.loss_func = loss_func



    # Metrics
    def Loss_Low(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [0,0])
    
    def Loss_Moderate(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [0,1])
    
    def Loss_Elevated(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [0,2])
    
    def Loss_High(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [0,3])
    
    
    # Metrics retrieval function
    def get_metrics(self) -> list:
        return [
                self.Loss_Low, 
                self.Loss_Moderate, 
                self.Loss_Elevated, 
                self.Loss_High
            ]

# %% ../nbs/metrics.ipynb 10
class GEODSTAPMetrics(RegressiveMetrics):
    def __init__(self, loss_func, indices:str='geodstap'):
        super().__init__(loss_func)
        self.indices = indices
        
        
    # Metrics
    def Loss_Low(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [0,0])
    
    def Loss_Medium(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [0,1])
    
    def Loss_Active(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [0,2])
    
    def Loss_G0(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [1,0])
    
    def Loss_G1(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [1,1])
    
    def Loss_G2(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [1,2])
        
    def Loss_G3(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [1,3])
    
    def Loss_G4(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [1,4])
    
    def Loss_G5(self, input, target):
        return self._apply_weighted_loss_by_level(input, target, [1,5])
    

    # Metrics retrieval function
    def get_metrics(self) -> list:
        if self.indices == 'geodst':
            return [
                self.Loss_G0, 
                self.Loss_G1, 
                self.Loss_G2, 
                self.Loss_G3, 
                self.Loss_G4, 
                self.Loss_G5
            ]
        
        elif self.indices == 'geoap':
            return [
                    self.Loss_Low, 
                    self.Loss_Medium, 
                    self.Loss_Active
                ]
        
        return [
                self.Loss_Low, 
                self.Loss_Medium, 
                self.Loss_Active,
                self.Loss_G0, 
                self.Loss_G1, 
                self.Loss_G2, 
                self.Loss_G3, 
                self.Loss_G4, 
                self.Loss_G5
            ]

# %% ../nbs/metrics.ipynb 12
class ClassificationMetrics(Metrics):
    def __init__(self, loss_func):
        super().__init__()
        self.loss_func = loss_func



    def _compute_misclassifications(self, predictions, targets):
        # Use the weighted loss tensor from the provided loss function
        classifier = self.loss_func.weighted_loss_tensor
        
        # Get the true and predicted labels using the classifier
        true_labels = classifier(targets)
        predicted_labels = classifier(predictions)

        # Misclassifications are those where the predicted label does not match the true label
        misclassified_labels = (true_labels != predicted_labels).int() * predicted_labels

        return misclassified_labels

    def _count_misclassifications_by_position(self, predictions, targets, row, col):
        # Calculate misclassifications for a specific (row, column) pair
        misclassified_labels = self._compute_misclassifications(predictions, targets)
        
        # Extract the specific misclassification at the (row, column) position and sum across the time dimension
        if row < misclassified_labels.shape[1] and col < misclassified_labels.shape[2]:
            misclassification_count = misclassified_labels[:, row, col].sum().item()
        else:
            misclassification_count = 0  # Out of bounds, assume no misclassification
        
        return misclassification_count
  
    
    @abstractmethod
    def get_metrics(self) -> list:
        return NotImplementedError

# %% ../nbs/metrics.ipynb 14
class SOLFMYClassificationMetrics(ClassificationMetrics):
    def __init__(self, loss_func):
        super().__init__(loss_func)


    # Metrics
    def Missclassifications_Low(self, predictions, targets):
        return self._count_misclassifications_by_position(predictions, targets, 0, 1)

    def Missclassifications_Moderate(self, predictions, targets):
        return self._count_misclassifications_by_position(predictions, targets, 0, 2)

    def Missclassifications_Elevated(self, predictions, targets):
        return self._count_misclassifications_by_position(predictions, targets, 0, 3)

    def Missclassifications_High(self, predictions, targets):
        return self._count_misclassifications_by_position(predictions, targets, 0, 4)


    # Metrics retrieval function
    def get_metrics(self) -> list:
        return [
                self.Missclassifications_Low,
                self.Missclassifications_Moderate, 
                self.Missclassifications_Elevated, 
                self.Missclassifications_High
            ]

# %% ../nbs/metrics.ipynb 16
class GEODSTAPClassificationMetrics(ClassificationMetrics):
    def __init__(self, loss_func, indices:str='geodstap'):
        super().__init__(loss_func)
        self.indices = indices


    # Metrics
    def Missclassifications_Low(self, predictions, targets):
        return self._count_misclassifications_by_position(predictions, targets, 0, 1)

    def Missclassifications_Medium(self, predictions, targets):
        return self._count_misclassifications_by_position(predictions, targets, 0, 2)

    def Missclassifications_Active(self, predictions, targets):
        return self._count_misclassifications_by_position(predictions, targets, 0, 3)

    def Missclassifications_G0(self, predictions, targets):
        return self._count_misclassifications_by_position(predictions, targets, 1, 1)

    def Missclassifications_G1(self, predictions, targets):
        return self._count_misclassifications_by_position(predictions, targets, 1, 2)

    def Missclassifications_G2(self, predictions, targets):
        return self._count_misclassifications_by_position(predictions, targets, 1, 3)

    def Missclassifications_G3(self, predictions, targets):
        return self._count_misclassifications_by_position(predictions, targets, 1, 4)

    def Missclassifications_G4(self, predictions, targets):
        return self._count_misclassifications_by_position(predictions, targets, 1, 5)

    def Missclassifications_G5(self, predictions, targets):
        return self._count_misclassifications_by_position(predictions, targets, 1, 6)

    # Metrics retrieval function
    def get_metrics(self) -> list:
        if self.indices == 'geodst':
            return [
                    self.Missclassifications_G0,
                    self.Missclassifications_G1, 
                    self.Missclassifications_G2, 
                    self.Missclassifications_G3, 
                    self.Missclassifications_G4, 
                    self.Missclassifications_G5
                ]
        
        elif self.indices == 'geoap':
            return [
                    self.Missclassifications_Low, 
                    self.Missclassifications_Medium, 
                    self.Missclassifications_Active
                ]
        
        return [
                self.Missclassifications_Low, 
                self.Missclassifications_Medium, 
                self.Missclassifications_Active, 
                self.Missclassifications_G0, 
                self.Missclassifications_G1,
                self.Missclassifications_G2,
                self.Missclassifications_G3,
                self.Missclassifications_G4,
                self.Missclassifications_G5
            ]

# %% ../nbs/metrics.ipynb 18
class LossMetrics(Metrics):
    def __init__(self, loss_func, indices:str = ''):
        super().__init__()
        self.loss_func = loss_func
        self.indices = indices

    ## Metrics Not Available
    def Metrics_Not_Available(self, input, target): return np.nan 
    
    # Metrics retrieval
    def get_metrics(self):
        if isinstance(self.loss_func, ClassificationLoss):
            if self.indices.lower() == 'solfsmy':
                return SOLFMYClassificationMetrics(self.loss_func).get_metrics()
            if self.indices.lower() in ['geodstap', 'geoap', 'geodst']:
                return GEODSTAPClassificationMetrics(self.loss_func, self.indices).get_metrics()
        
        if isinstance(self.loss_func, WeightedLoss):
            if self.indices.lower() == 'solfsmy':
                return SOLFMYMetrics(self.loss_func).get_metrics()
            
            if self.indices.lower() in ['geodstap', 'geoap', 'geodst']:
                return GEODSTAPMetrics(self.loss_func, self.indices).get_metrics()
        
        return [self.Metrics_Not_Available]

# %% ../nbs/metrics.ipynb 22
class OutlierDetectionMetrics(Metrics):
    def __init__(self, threshold=3.5):
        super().__init__()
        self.threshold = threshold

    @staticmethod
    def _modified_z_score(x):
        """
        Calculate the Modified Z-Score for each variable in the tensor.
        
        Parameters:
        tensor (torch.Tensor): Input tensor of shape (batch_size, variables, horizon)
        
        Returns:
        torch.Tensor: Modified Z-Score tensor of the same shape as input
        """
        median = torch.median(x, dim=2, keepdim=True).values
        
        mad = torch.median(torch.abs(x - median), dim=2, keepdim=True).values
        mad = torch.where(mad == 0, torch.tensor(1.0, device=x.device), mad)
        
        modified_z_scores = 0.6745 * (x - median) / mad
        
        return modified_z_scores

    def _detect_outliers(self, values):
        """
        Detect outliers based on Modified Z-Scores.
        
        Parameters:
        z_scores (torch.Tensor): Modified Z-Scores tensor
        
        Returns:
        torch.Tensor: Boolean tensor indicating outliers
        """
        z_scores = self._modified_z_score(values)
        return torch.abs(z_scores) > self.threshold
    
    def _evaluate_outlier_predicted(self, y_true, y_pred):
        """
        Evaluate the performance of outlier detection.
        
        Parameters:
        y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)
        y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true
        
        Returns:
        AttrDict: Dictionary with true/false positives, false negatives, indices of true/predicted outliers
        """    
        # Detect outliers based on the threshold
        true_outliers = self._detect_outliers(y_true)
        pred_outliers = self._detect_outliers(y_pred)
        
        # Evaluate the detection by comparing true outliers and predicted outliers
        tp = torch.sum((pred_outliers & true_outliers).float())  # True Positives
        fp = torch.sum((pred_outliers & ~true_outliers).float()) # False Positives
        fn = torch.sum((~pred_outliers & true_outliers).float()) # False Negatives
        tn = torch.sum((~pred_outliers & ~true_outliers).float()) # True Negatives

        return AttrDict({
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "true_outliers": true_outliers,
            "predicted_outliers": pred_outliers
        })
    
    
    @abstractmethod
    def get_metrics(self) -> list:
        return NotImplementedError

# %% ../nbs/metrics.ipynb 24
class F1ScoreMetrics(OutlierDetectionMetrics):
    def __init__(self, threshold=3.5, metrics='F1_Score'):
        super().__init__(threshold)
        self.metrics = metrics
    

    # Metrics
    def Precision(self, y_true, y_pred):
        stats = self._evaluate_outlier_predicted(y_true, y_pred)

        # To avoid divide by 0
        if (stats.tp + stats.fp) > 0:
            precision = stats.tp / (stats.tp + stats.fp)  
        else: 
            precision = torch.tensor(0.0)

        return precision
    
    def Recall(self, y_true, y_pred):
        stats = self._evaluate_outlier_predicted(y_true, y_pred)

        if (stats.tp + stats.fn) > 0:
            recall = stats.tp / (stats.tp + stats.fn)
        else: 
            recall = torch.tensor(0.0)

        return recall
    
    def F1_Score(self, y_true, y_pred):
        precision = self.Precision(y_true, y_pred)
        recall = self.Recall(y_true, y_pred)

        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else: 
            f1_score = torch.tensor(0.0)

        return f1_score
    
    def Accuracy_Score(self, y_true, y_pred):
        stats = self._evaluate_outlier_predicted(y_true, y_pred)
        
        if (stats.tp + stats.fp + stats.fn + stats.tn) > 0:
            return (stats.tp + stats.tn) / (stats.tp + stats.fp + stats.fn + stats.tn)
        else:
            return torch.tensor(0.0)
    
    def Specificity(self, y_true, y_pred):
        stats = self._evaluate_outlier_predicted(y_true, y_pred)
        
        if (stats.tn + stats.fp) > 0:
            return stats.tn / (stats.tn + stats.fp)
        else:
            return torch.tensor(0.0)

    def Negative_Predictive_Value(self, y_true, y_pred):
        stats = self._evaluate_outlier_predicted(y_true, y_pred)

        if (stats.tn + stats.fn) > 0:
            return stats.tn / (stats.tn + stats.fn)
        else:
            return torch.tensor(0.0)
    
    def Δ_Detected_Outliers (self, y_true, y_pred):
        stats = self._evaluate_outlier_predicted(y_true, y_pred)
        
        return torch.sum(stats.true_outliers & ~stats.predicted_outliers)


    # Metrics retrieval function
    def get_metrics(self) -> list:
        if self.metrics == 'F1_Score':
            return [self.F1_Score]
        elif self.metrics == 'All':
            return [self.Precision, self.Recall, self.F1_Score, self.Accuracy_Score, self.Specificity, self.Negative_Predictive_Value, self.Δ_Detected_Outliers ]
        else:
            return [self.Precision, self.Recall, self.F1_Score, self.Δ_Detected_Outliers ]

# %% ../nbs/metrics.ipynb 26
class AUPRCMetric(OutlierDetectionMetrics):
    def __init__(self, threshold=3.5):
        super().__init__(threshold)


    # Metrics
    def AURPC(self, y_true, y_pred):
        """
        Calculate the Area Under the Precision-Recall Curve (AUPRC).
        
        Parameters:
        y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)
        y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true
        
        Returns:
        torch.Tensor: AUPRC score
        """
        pred_z_scores = self._modified_z_score(y_pred)
        
        pred_z_scores_flat = pred_z_scores.view(-1).cpu().numpy()
        true_outliers_flat = self._detect_outliers(y_true).view(-1).cpu().numpy()
        
        # Use precision_recall_curve to get precision and recall for different thresholds
        precision, recall, _ = precision_recall_curve(true_outliers_flat, pred_z_scores_flat)
        
        auprc_value = auc(recall, precision)
        
        return torch.tensor(auprc_value, device=y_true.device)
    

    # Metrics retrieval function
    def get_metrics(self) -> list:
        return [self.AURPC]

# %% ../nbs/metrics.ipynb 28
class KSDifferenceMetric(Metrics):
    def __init__(self, threshold=3.5):
        super().__init__()
        self.threshold = threshold

    @staticmethod
    def skewness(x):
        mean = torch.mean(x)
        std_dev = torch.std(x, unbiased=False)
        
        skewness = torch.mean(((x - mean) / std_dev) ** 3)
        return skewness

    @staticmethod
    def kurtosis(x):
        mean = torch.mean(x)
        std_dev = torch.std(x, unbiased=False)
        
        kurtosis = torch.mean(((x - mean) / std_dev) ** 4)
        return kurtosis
    

    # Metrics
    def Δ_Skewness(self, y_true, y_pred):
        true_skewness = KSDifferenceMetric.skewness(y_true)
        pred_skewness = KSDifferenceMetric.skewness(y_pred)
        
        return torch.abs(true_skewness - pred_skewness)
    
    def Δ_Kurtosis(self, y_true, y_pred):
        true_kurtosis = KSDifferenceMetric.kurtosis(y_true)
        pred_kurtosis = KSDifferenceMetric.kurtosis(y_pred)
        
        return torch.abs(true_kurtosis - pred_kurtosis)
        

    # Metrics retrieval function
    def get_metrics(self) -> list:
        return [self.Δ_Skewness, self.Δ_Kurtosis]

# %% ../nbs/metrics.ipynb 31
class AssociationMetrics(Metrics):
    def __init__(self):
        super().__init__()

    # Metrics
    def R_Correlation(self, y_true, y_pred):
        """
        Calculate the Pearson Correlation Coefficient (R Correlation) between true and predicted values.
        
        Parameters:
        y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)
        y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true
        
        Returns:
        torch.Tensor: R Correlation coefficient
        """
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)
        
        # To be able to use torch.corrcoef, we need to stack the tensors
        stacked = torch.stack([y_true_flat, y_pred_flat])
        
        corr_matrix = torch.corrcoef(stacked)
        
        r_value = corr_matrix[0, 1]
        return r_value

    def R2_Score(self, y_true, y_pred):
        """
        Calculate the R^2 score.
        
        Parameters:
        y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)
        y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true
        
        Returns:
        torch.Tensor: R^2 score
        """
        y_true_mean = torch.mean(y_true, dim=2, keepdim=True)
        print(y_true_mean.shape, y_true.shape)
        
        # Total Sum of Squares
        ss_tot = torch.sum((y_true - y_true_mean) ** 2)
        
        # Residual Sum of Squares
        ss_res = torch.sum((y_true - y_pred) ** 2)
        
        r2 = 1 - ss_res / ss_tot
        
        return r2
    
    
    # Metrics retrieval function
    def get_metrics(self) -> list:
        return [self.R_Correlation, self.R2_Score]
    
    

# %% ../nbs/metrics.ipynb 34
class AccuracyMetrics(Metrics):
    def __init__(self):
        super().__init__()


    # Metrics
    def sMAPE(self, y_true, y_pred):
        """
        Calculate the Symmetric Mean Absolute Percentage Error (sMAPE).
        
        Parameters:
        y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)
        y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true
        
        Returns:
        torch.Tensor: sMAPE value
        """
        epsilon = 1e-8  # small constant to prevent division by zero
        abs_error = torch.abs(y_true - y_pred)
        symetric_error = ((torch.abs(y_true) + torch.abs(y_pred)) / 2.0) + epsilon
        
        smape = torch.mean(abs_error / symetric_error) * 100
        return smape
    
    def MSA(self, y_true, y_pred):
        """
        Calculate the Mean Scaled Absolute Error (MSA).
        
        Parameters:
        y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)
        y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true
        
        Returns:
        torch.Tensor: MSA value
        """
        # Calculate the natural logarithm of the ratio
        log_ratio = torch.abs(torch.log(y_pred / y_true))

        msa = (torch.exp(torch.median(log_ratio)) - 1) * 100
        
        return msa
    

    # Metrics retrieval function
    def get_metrics(self) -> list:
        return [self.sMAPE, self.MSA]


# %% ../nbs/metrics.ipynb 37
class BiasMetrics(Metrics):
    def __init__(self):
        super().__init__()

    # Metrics
    def SSPB(self, y_true, y_pred):
        """
        Calculate the Symmetric Signed Percentage Bias (SSPB).
        
        Parameters:
        y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)
        y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true
        
        Returns:
        torch.Tensor: SSPB value
        """
        log_ratio = torch.log(y_pred / y_true)
        median_log_ratio = torch.median(log_ratio)

        sign = torch.sign(median_log_ratio)
        
        return sign * (torch.exp(torch.abs(median_log_ratio)) - 1) * 100


    # Metrics retrieval function
    def get_metrics(self) -> list:
        return [self.SSPB]
