# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/metrics.ipynb.

# %% auto 0
__all__ = ['Metrics', 'RegressiveMetrics', 'SOLFMYMetrics', 'GEODSTAPMetrics', 'ClassificationMetrics',
           'SOLFMYClassificationMetrics', 'GEODSTAPClassificationMetrics', 'LossMetrics', 'OutlierDetectionMetrics',
           'F1ScoreMetrics', 'AUPRCMetric', 'KSDifferenceMetric', 'AssociationMetrics', 'AccuracyMetrics',
           'BiasMetrics', 'ValidationMetricsHandler']

# %% ../nbs/metrics.ipynb 0
import sys
sys.path.append('..')
from abc import ABC, abstractmethod
import torch
import numpy as np
import pandas as pd
from tsai.basics import *
from .losses import wMAELoss, MSELoss, WeightedLoss, ClassificationLoss
from sklearn.metrics import precision_recall_curve, auc
from optuna.study import StudyDirection




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
        """
        <p>Calculate the precision metric, which measures the ratio of correctly predicted positive observations to the total predicted positives.</p>

        <h3>Parameters:</h3>
        <ul>
            <li>y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)</li>
            <li>y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true</li>
        </ul>

        <h3>Returns:</h3>
        <p>torch.Tensor: Precision score<p>
        """
        stats = self._evaluate_outlier_predicted(y_true, y_pred)

        # To avoid divide by 0
        if (stats.tp + stats.fp) > 0:
            precision = stats.tp / (stats.tp + stats.fp)  
        else: 
            precision = torch.tensor(0.0)

        return precision

    
    def Recall(self, y_true, y_pred):
        """
        <p>Calculate the recall metric, which measures the ratio of correctly predicted positive observations to all observations in the actual class.</p>

        <h3>Parameters:</h3>
        <ul>
            <li>y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)</li>
            <li>y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true</li>
        </ul>

        <h3>Returns:</h3>
        <p>torch.Tensor: Recall score<p>
        """
        stats = self._evaluate_outlier_predicted(y_true, y_pred)

        if (stats.tp + stats.fn) > 0:
            recall = stats.tp / (stats.tp + stats.fn)
        else: 
            recall = torch.tensor(0.0)

        return recall

    
    def F1_Score(self, y_true, y_pred):
        """
        <p>Calculate the F1 score, which is the harmonic mean of precision and recall. It is used as a measure of a model’s accuracy on a dataset.</p>

        <h3>Parameters:</h3>
        <ul>
            <li>y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)</li>
            <li>y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true</li>
        </ul>

        <h3>Returns:</h3>
        <p>torch.Tensor: F1 score<p>
        """
        precision = self.Precision(y_true, y_pred)
        recall = self.Recall(y_true, y_pred)

        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else: 
            f1_score = torch.tensor(0.0)

        return f1_score

    
    def Accuracy_Score(self, y_true, y_pred):
        """
        <p>Calculate the accuracy score, which is the ratio of correctly predicted observations to the total observations.</p>

        <h3>Parameters:</h3>
        <ul>
            <li>y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)</li>
            <li>y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true</li>
        </ul>

        <h3>Returns:</h3>
        <p>torch.Tensor: Accuracy score<p>
        """
        stats = self._evaluate_outlier_predicted(y_true, y_pred)
            
        if (stats.tp + stats.fp + stats.fn + stats.tn) > 0:
            return (stats.tp + stats.tn) / (stats.tp + stats.fp + stats.fn + stats.tn)
        else:
            return torch.tensor(0.0)

    
    def Specificity(self, y_true, y_pred):
        """
        <p>Calculate the specificity metric, which measures the proportion of true negatives that are correctly identified.</p>

        <h3>Parameters:</h3>
        <ul>
            <li>y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)</li>
            <li>y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true</li>
        </ul>

        <h3>Returns:</h3>
        <p>torch.Tensor: Specificity score<p>
        """
        stats = self._evaluate_outlier_predicted(y_true, y_pred)
            
        if (stats.tn + stats.fp) > 0:
            return stats.tn / (stats.tn + stats.fp)
        else:
            return torch.tensor(0.0)


    def Negative_Predictive_Value(self, y_true, y_pred):
        """
        <p>Calculate the Negative Predictive Value (NPV), which measures the proportion of true negatives among all negative predictions.</p>

        <h3>Parameters:</h3>
        <ul>
            <li>y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)</li>
            <li>y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true</li>
        </ul>

        <h3>Returns:</h3>
        <p>torch.Tensor: Negative Predictive Value score<p>
        """
        stats = self._evaluate_outlier_predicted(y_true, y_pred)

        if (stats.tn + stats.fn) > 0:
            return stats.tn / (stats.tn + stats.fn)
        else:
            return torch.tensor(0.0)

    
    def Detected_Outliers_Difference (self, y_true, y_pred):
        """
        <p>Calculate the change in detected outliers (Δ Detected Outliers), representing the number of true outliers not predicted as outliers.</p>

        <h3>Parameters:</h3>
        <ul>
            <li>y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)</li>
            <li>y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true</li>
        </ul>

        <h3>Returns:</h3>
        <p>torch.Tensor: Count of undetected outliers<p>
        """    
        stats = self._evaluate_outlier_predicted(y_true, y_pred)
        
        return torch.sum(stats.true_outliers & ~stats.predicted_outliers)


    # Metrics retrieval function
    def get_metrics(self) -> list:
        if self.metrics == 'F1_Score':
            return [self.F1_Score]
        elif self.metrics == 'All':
            return [self.Precision, self.Recall, self.F1_Score, self.Accuracy_Score, self.Specificity, self.Negative_Predictive_Value, self.Detected_Outliers_Difference ]
        else:
            return [self.Precision, self.Recall, self.F1_Score, self.Detected_Outliers_Difference ]

# %% ../nbs/metrics.ipynb 26
class AUPRCMetric(OutlierDetectionMetrics):
    def __init__(self, threshold=3.5):
        super().__init__(threshold)


    # Metrics
    def AURPC(self, y_true, y_pred):
        """
        <p>Calculate the Area Under the Precision-Recall Curve (AUPRC), a 
        metric used to evaluate the effectiveness of a model in identifying rare, important events (outliers)</p>
        
        <h3>Parameters:</h3>
        <ul>
            <li>y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)</li>
            <li>y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true</li>
        
        <h3>Returns:</h3>
        <p>torch.Tensor: AUPRC score<p>
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
        # As batches are randomly generated and each variable is independent
        mean = torch.mean(x, dim=2, keepdim=True)
        std_dev = torch.std(x, dim=2, unbiased=True, keepdim=True)
        
        skewness = torch.mean(((x - mean) / std_dev) ** 3, dim=2)
        return skewness

    @staticmethod
    def kurtosis(x):
        mean = torch.mean(x, dim=2, keepdim=True)
        std_dev = torch.std(x, dim=2, unbiased=True, keepdim=True)
        
        kurtosis = torch.mean(((x - mean) / std_dev) ** 4, dim=2)
        return kurtosis
    

    # Metrics
    def Skewness_Difference(self, y_true, y_pred):
        """
        <p>Calculate the absolute difference in skewness between the actual and predicted values, which measures the asymmetry of the data distribution.</p>

        <h3>Parameters:</h3>
        <ul>
            <li>y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)</li>
            <li>y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true</li>
        </ul>

        <h3>Returns:</h3>
        <p>torch.Tensor: Absolute difference in skewness between y_true and y_pred<p>
        """
        true_skewness = KSDifferenceMetric.skewness(y_true)
        pred_skewness = KSDifferenceMetric.skewness(y_pred)
        
        return torch.mean(torch.abs(true_skewness - pred_skewness), dim=[0, 1])

    
    def Kurtosis_Difference(self, y_true, y_pred):
        """
        <p>Calculate the absolute difference in kurtosis between the actual and predicted values, which measures the tailedness of the data distribution.</p>

        <h3>Parameters:</h3>
        <ul>
            <li>y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)</li>
            <li>y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true</li>
        </ul>

        <h3>Returns:</h3>
        <p>torch.Tensor: Absolute difference in kurtosis between y_true and y_pred<p>
        """
        true_kurtosis = KSDifferenceMetric.kurtosis(y_true)
        pred_kurtosis = KSDifferenceMetric.kurtosis(y_pred)
        
        return torch.mean(torch.abs(true_kurtosis - pred_kurtosis), dim=[0, 1])


    # Metrics retrieval function
    def get_metrics(self) -> list:
        return [self.Skewness_Difference, self.Kurtosis_Difference]

# %% ../nbs/metrics.ipynb 31
class AssociationMetrics(Metrics):
    def __init__(self):
        super().__init__()

    # Metrics
    def R_Correlation(self, y_true, y_pred):
        """
        <p>Calculate the Pearson Correlation Coefficient (R Correlation) between true and predicted values.</p>
        
        <h3>Parameters:</h3>
        <ul>
            <li>y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)</li>
            <li>y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true</li>
        </ul>
        
        <h3>Returns:</h3>
        <p>torch.Tensor: R Correlation coefficient<p>
        """
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        
        # To be able to use torch.corrcoef, we need to stack the tensors
        stacked = torch.stack([y_true_flat, y_pred_flat])
        
        corr_matrix = torch.corrcoef(stacked)
        
        r_value = corr_matrix[0, 1]
        return r_value


    def R2_Score(self, y_true, y_pred):
        """
        <p>Calculate the R^2 score, which measures the proportion of the variance in the dependent variable that is predictable from the independent variable(s).</p>
        
        <h3>Parameters:</h3>
        <ul>
            <li>y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)</li>
            <li>y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true</li>
        </ul>
        
        <h3>Returns:</h3>
        <p>torch.Tensor: R^2 score<p>
        """
        y_true_mean = torch.mean(y_true, dim=2, keepdim=True)
        
        # Total Sum of Squares
        ss_tot = torch.sum((y_true - y_true_mean) ** 2)
        
        # Residual Sum of Squares
        ss_res = torch.sum((y_true - y_pred) ** 2)
        
        if ss_tot == 0:
            r2 = torch.tensor(0.0)
        else:
            r2 = 1 - (ss_res / ss_tot)
        
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
        Calculate the Median Symmetric Accuracy (MSA).
        
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
        <p>Calculate the Symmetric Signed Percentage Bias (SSPB), which measures the percentage bias with consideration for the direction of the bias.</p>
        
        <h3>Parameters:</h3>
        <ul>
            <li>y_true (torch.Tensor): Actual values tensor of shape (batch_size, variables, horizon)</li>
            <li>y_pred (torch.Tensor): Predicted values tensor of the same shape as y_true</li>
        </ul>
        
        <h3>Returns:</h3>
        <p>torch.Tensor: SSPB value<p>
        """
        log_ratio = torch.log(y_pred / y_true)
        median_log_ratio = torch.median(log_ratio)

        sign = torch.sign(median_log_ratio)
        
        return sign * (torch.exp(torch.abs(median_log_ratio)) - 1) * 100



    # Metrics retrieval function
    def get_metrics(self) -> list:
        return [self.SSPB]

# %% ../nbs/metrics.ipynb 39
class ValidationMetricsHandler:
    """
    <p>A class to manage validation metrics for model evaluation. It allows listing available metrics, uploading requested metrics, and retrieving study directions and objective values.</p>
    
    <h3>Attributes:</h3>
    <ul>
        <li>available_metrics (list)[<i>Static</i>]: A list of available metrics provided by different metric classes.</li>
        <li>study_directions (dict)[<i>Static</i>]: A dictionary mapping metrics to their respective optimization directions (maximize or minimize).</li>
        <li>requested_metrics (dict): A dictionary storing metrics that have been requested for evaluation.</li>
    </ul>
    """
    
    available_metrics = [
        *F1ScoreMetrics(metrics='All').get_metrics(),
        *AUPRCMetric().get_metrics(),
        *KSDifferenceMetric().get_metrics(),
        *AssociationMetrics().get_metrics(),
        *AccuracyMetrics().get_metrics(),
        *BiasMetrics().get_metrics()
    ]

    study_directions = {
        'precision': StudyDirection.MAXIMIZE,                    # Higher precision is better (Range: [0, 1])
        'recall': StudyDirection.MAXIMIZE,                       # Higher recall is better (Range: [0, 1])
        'f1_score': StudyDirection.MAXIMIZE,                     # Higher F1 score is better (Range: [0, 1])
        'accuracy_score': StudyDirection.MAXIMIZE,               # Higher accuracy is better (Range: [0, 1])
        'specificity': StudyDirection.MAXIMIZE,                  # Higher specificity is better (Range: [0, 1])
        'negative_predictive_value': StudyDirection.MAXIMIZE,    # Higher NPV is better (Range: [0, 1])
        'detected_outliers_difference': StudyDirection.MINIMIZE, # Minimize the difference in detected outliers (Range: [0, ∞))
        'aurpc': StudyDirection.MAXIMIZE,                        # Higher AUPRC is better (Range: [0, 1])
        'skewness_difference': StudyDirection.MINIMIZE,          # Minimize skewness difference to target (Range: [−∞, ∞])
        'kurtosis_difference': StudyDirection.MINIMIZE,          # Minimize kurtosis difference to target (Range: [−∞, ∞])
        'r_correlation': StudyDirection.MAXIMIZE,                # Higher Pearson correlation is better (Range: [−1, 1])
        'r2_score': StudyDirection.MAXIMIZE,                     # Higher R² is better (Range: [−∞, 1])
        'smape': StudyDirection.MINIMIZE,                        # Lower SMAPE is better (Range: [0, ∞))
        'msa': StudyDirection.MAXIMIZE,                          # Higher MSA is better (Range: [0, 1])
        'sspb': StudyDirection.MINIMIZE                          # Minimize absolute SSPB (optimize for bias close to zero) (Range: [−100%, 100%])
    }


    def __init__(self, metrics:list):
        self.requested_metrics = {}
        self.add(metrics)

    @classmethod
    def list(cls):
        """
        <p>Display a list of available metrics along with their descriptions in a table format.</p>
        """
        table_rows = []
        
        for metric in cls.available_metrics:
            doc_html = metric.__doc__.strip().replace("\n", " ")
            table_rows.append(f"<tr><td style='text-align: left;'><strong>{metric.__name__}</strong></td><td style='text-align: left;'>{doc_html}</td></tr>")
        
        table_html = f"""
        <table>
            <thead>
                <tr>
                    <th style='text-align: left;'>Metric Name</th>
                    <th style='text-align: left;'>Description</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
        """
        
        display(HTML(table_html))


    # Request of metrics
    def add(self, metrics:list):
        """
        <p>Upload a list of metrics to the factory for evaluation. The metrics are converted to lowercase for consistency.</p>
        
        <h3>Parameters:</h3>
        <ul>
            <li>metrics (list): A list of metric names to be uploaded for evaluation.</li>
        </ul>
        
        <h3>Raises:</h3>
        <p>ValueError: If any metric in the provided list is not found in the available metrics.</p>
        """
        metrics = [metric.lower() for metric in metrics]

        for metric in ValidationMetricsHandler.available_metrics:
            metric_name = metric.__name__.lower()
            if metric_name in metrics:
                self.requested_metrics[metric_name] = metric
                metrics.remove(metric.__name__.lower())
        
        if len(metrics) > 0:
            raise ValueError(f"Metrics not found: {metrics}. Please use ValidationMetricsFactory.list() to see available metrics.")
        
    def remove(self, metrics:list):
        """
        <p>Remove a list of metrics from the factory that were previously uploaded for evaluation. The metrics are converted to lowercase for consistency.</p>
        
        <h3>Parameters:</h3>
        <ul>
            <li>metrics (list): A list of metric names to be removed from evaluation.</li>
        </ul>
        
        <h3>Raises:</h3>
        <p>ValueError: If any metric in the provided list is not found in the requested metrics.</p>
        """
        metrics = [metric.lower() for metric in metrics]

        for metric in metrics:
            if metric in self.requested_metrics:
                self.requested_metrics.pop(metric)
            else:
                raise ValueError(f"Metric not found: {metric}. Please use ValidationMetricsFactory.get_metrics() to see requested metrics.")
        

    # Creation functions
    def get_metrics(self) -> list:
        """
        <p>Retrieve the list of requested metrics for evaluation.</p>
        
        <h3>Returns:</h3>
        <p>list: A list of requested metric objects.</p>
        """
        return list(self.requested_metrics.values())

    def get_study_directions(self) -> list:    
        """
        <p>Retrieve the study directions (maximize or minimize) for the requested metrics.</p>
        
        <h3>Returns:</h3>
        <p>list: A list of study directions corresponding to the requested metrics.</p>
        """
        return [self.study_directions[metric] for metric in self.requested_metrics.keys()]
    
    def get_objective_values(self, metrics_results:List[AvgMetric]) -> list:
        """
        <p>Extract the objective values from the results of the requested metrics.</p>
        
        <h3>Parameters:</h3>
        <ul>
            <li>metrics_results (List[AvgMetric]): A list of metric result objects from which to extract values.</li>
        </ul>
        
        <h3>Returns:</h3>
        <p>list: A list of metric values extracted from the provided results.</p>
        """
        object_values = []
        for metric, requested_metric in zip(metrics_results, self.requested_metrics.keys()):
            metric_name = metric.name.lower()

            if metric_name == requested_metric:
                # As SSPB could be positive or negative, but the better is to be closer to 0
                if metric_name == 'sspb':
                    object_values.append(np.abs(metric.value)) 
                else:
                    object_values.append(metric.value)
            else:
                raise ValueError(f"Unexpected metric found: {metric_name}. Expected: {requested_metric}")
            
            
        return (metric_result.value for metric_result in metrics_results)
    

    @classmethod
    def are_best_values(cls, main_metric: str, best_values, trial_values) -> bool:
        """
        <p>Determine if the trial values are better than the current best values for a given metric.</p>
        
        <h3>Parameters:</h3>
        <ul>
            <li><b>main_metric</b> (str): The name of the main metric used for comparison.</li>
            <li><b>best_values</b> (list): A list of current best values for various metrics.</li>
            <li><b>trial_values</b> (list): A list of new trial values to compare against the best values.</li>
        </ul>
        
        <h3>Returns:</h3>
        <p>bool: True if the trial values are overall better than the best values, False otherwise.</p>
        """
        main_metric = main_metric.lower()

        if best_values is None:
            return True

        if len(best_values) == 1:
            best_value = best_values[0].value
            trial_value = trial_values[0].value
            direction = cls.study_directions.get(main_metric)

            if direction == StudyDirection.MAXIMIZE:
                return trial_value > best_value
            elif direction == StudyDirection.MINIMIZE:
                return trial_value < best_value
            return False

        improvement_count = 0
        num_metrics_to_compare = len(best_values) // 2
        for best_metric, trial_metric in zip(best_values, trial_values):
            metric_name = best_metric.name.lower()
            direction = cls.study_directions.get(metric_name)

            if direction == StudyDirection.MAXIMIZE and trial_metric.value > best_metric.value:
                if main_metric == metric_name:
                    improvement_count += num_metrics_to_compare
                else:
                    improvement_count += 1
            elif direction == StudyDirection.MINIMIZE and trial_metric.value < best_metric.value:
                if main_metric == metric_name:
                    improvement_count += num_metrics_to_compare
                else:
                    improvement_count += 1

        return improvement_count > num_metrics_to_compare
            

