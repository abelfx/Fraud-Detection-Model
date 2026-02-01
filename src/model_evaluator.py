import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.logger import setup_logger

logger = setup_logger(__name__, "model_evaluator.log")


class ModelEvaluator:
    """Handles evaluation and scoring of fraud detection models."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def evaluate(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        y_pred: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of model predictions.
        """
        logger.info("Evaluating model performance")
        
        # Generate predictions if not provided
        if y_pred is None:
            y_pred = (y_pred_proba[:, 1] >= self.threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba[:, 1]),
            'average_precision': average_precision_score(y_true, y_pred_proba[:, 1]),
            'threshold': self.threshold
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Calculate specificity (True Negative Rate)
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Log metrics
        logger.info("\n" + "="*60)
        logger.info("EVALUATION METRICS")
        logger.info("="*60)
        logger.info(f"Accuracy:           {metrics['accuracy']:.4f}")
        logger.info(f"Precision:          {metrics['precision']:.4f}")
        logger.info(f"Recall:             {metrics['recall']:.4f}")
        logger.info(f"F1 Score:           {metrics['f1_score']:.4f}")
        logger.info(f"ROC AUC:            {metrics['roc_auc']:.4f}")
        logger.info(f"AUPRC (AP):         {metrics['average_precision']:.4f} *** PRIMARY METRIC FOR IMBALANCED DATA ***")
        logger.info(f"Specificity:        {metrics['specificity']:.4f}")
        logger.info("="*60 + "\n")
        
        return metrics
    
    def find_optimal_threshold(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        metric: str = 'f1'
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal classification threshold.
        """
        logger.info(f"Finding optimal threshold based on {metric}")
        
        # Get precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(
            y_true, y_pred_proba[:, 1]
        )
        
        # Calculate F1 scores
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        # Find optimal threshold based on metric
        if metric == 'f1':
            optimal_idx = np.argmax(f1_scores)
        elif metric == 'precision':
            optimal_idx = np.argmax(precisions)
        elif metric == 'recall':
            optimal_idx = np.argmax(recalls)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Get threshold (handling edge case)
        if optimal_idx < len(thresholds):
            optimal_threshold = thresholds[optimal_idx]
        else:
            optimal_threshold = thresholds[-1]
        
        # Evaluate at optimal threshold
        y_pred_optimal = (y_pred_proba[:, 1] >= optimal_threshold).astype(int)
        
        metrics = {
            'threshold': optimal_threshold,
            'precision': precisions[optimal_idx],
            'recall': recalls[optimal_idx],
            'f1_score': f1_scores[optimal_idx]
        }
        
        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
        logger.info(f"Metrics at optimal threshold: {metrics}")
        
        return optimal_threshold, metrics
    

def evaluate_model(
    model_trainer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    optimize_threshold: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a trained model.
    """
    evaluator = ModelEvaluator()
    
    # Get predictions
    y_pred_proba = model_trainer.predict_proba(X_test)
    y_pred = model_trainer.predict(X_test)
    
    # Evaluate
    metrics = evaluator.evaluate(y_test, y_pred_proba, y_pred)
    
    # Find optimal threshold if requested
    if optimize_threshold:
        optimal_threshold, optimal_metrics = evaluator.find_optimal_threshold(
            y_test, y_pred_proba, metric='f1'
        )
        metrics['optimal_threshold'] = optimal_threshold
        metrics['optimal_metrics'] = optimal_metrics
    
    # Generate classification report
    report = evaluator.generate_classification_report(y_test, y_pred)
    metrics['classification_report'] = report
    
    return metrics
