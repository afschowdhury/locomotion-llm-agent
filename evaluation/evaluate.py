"""
File: evaluation/evaluate.py
Description: Evaluator class for computing locomotion prediction metrics.
"""

from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix


class Evaluator:
    """
    Evaluator for locomotion prediction performance.
    
    This class:
      - Accumulates ground truth and predicted labels.
      - Computes overall and per-command-type metrics.
    """
    
    def __init__(self):
        self.overall_ground_truth = []
        self.overall_predictions = []
        self.metrics_by_command_type = {}
        self.ltm_engagement_count = 0
        self.clarity_scores = []
        self.confidences = []
        self.confidence_correctness = []

    def add_sample(self, true_label, predicted_label, command_type = "clear",
                   refined = False, clarity_score = None, confidence = None):
        """
        Adds a single sample's results to the evaluator.
        
        Args:
            true_label (str): Ground truth locomotion mode.
            predicted_label (str): Predicted locomotion mode from the agent.
            command_type (str): Command category (e.g., "clear", "vague", "safety-critical").
            refined (bool): Flag indicating if refinement (LTM engagement) was performed.
            clarity_score (float, optional): Clarity score of the sample, if available.
            confidence (float, optional): Confidence value (0 to 1) for the prediction.
        """
        self.overall_ground_truth.append(true_label)
        self.overall_predictions.append(predicted_label)
        if refined:
            self.ltm_engagement_count += 1

        if clarity_score is not None:
            self.clarity_scores.append(clarity_score)

        if confidence is not None:
            self.confidences.append(confidence)
            self.confidence_correctness.append(1 if predicted_label == true_label else 0)

        command_type = command_type.lower()
        if command_type not in self.metrics_by_command_type:
            self.metrics_by_command_type[command_type] = {"y_true": [], "y_pred": []}
        self.metrics_by_command_type[command_type]["y_true"].append(true_label)
        self.metrics_by_command_type[command_type]["y_pred"].append(predicted_label)

    def get_overall_report(self):
        """
        Computes the overall classification report and confusion matrix.
        
        Returns:
            dict: A dictionary with "classification_report" and "confusion_matrix" keys.
        """
        report = classification_report(
            self.overall_ground_truth, self.overall_predictions, output_dict=True
        )
        conf_matrix = confusion_matrix(self.overall_ground_truth, self.overall_predictions)
        return {
            "classification_report": report,
            "confusion_matrix": conf_matrix
        }

    def get_command_type_reports(self):
        """
        Computes classification reports and confusion matrices for each command type.
        
        Returns:
            dict: A dictionary mapping each command type to its corresponding report and matrix.
        """
        reports = {}
        for command_type, data in self.metrics_by_command_type.items():
            clf_report = classification_report(
                data["y_true"], data["y_pred"], output_dict=True
            )
            conf_matrix = confusion_matrix(data["y_true"], data["y_pred"]).tolist()
            reports[command_type] = {
                "classification_report": clf_report,
                "confusion_matrix": conf_matrix
            }
        return reports

    def get_brier_score(self):
        """
        Computes the Brier Score.

        The Brier Score is calculated as (1/N) * Σ (p_i - y_i)^2, where p_i is the model's 
        confidence and y_i is 1 if the prediction is correct or 0 if not.
        
        Returns:
            float: The Brier Score, or None if no confidence data is recorded.
        """
        import numpy as np
        if not self.confidences or not self.confidence_correctness:
            return None
        confidences = np.array(self.confidences)
        correctness = np.array(self.confidence_correctness)
        return np.mean((confidences - correctness) ** 2)

    def get_ece(self, num_bins = 10):
        """
        Computes the Expected Calibration Error (ECE).

        ECE measures the weighted average difference between the average confidence 
        and average accuracy over multiple confidence bins.
        
        Args:
            num_bins (int): Number of bins to use for the calculation.
        
        Returns:
            float: The computed ECE value, or None if no confidence data is available.
        """
        import numpy as np
        if not self.confidences or not self.confidence_correctness:
            return None
        confidences = np.array(self.confidences)
        correctness = np.array(self.confidence_correctness)
    
        bins = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        n = len(confidences)
        for i in range(num_bins):
            bin_lower = bins[i]
            bin_upper = bins[i + 1]
            if i == 0:
                bin_mask = (confidences >= bin_lower) & (confidences <= bin_upper)
            else:
                bin_mask = (confidences > bin_lower) & (confidences <= bin_upper)
            if np.sum(bin_mask) > 0:
                avg_confidence = np.mean(confidences[bin_mask])
                avg_accuracy = np.mean(correctness[bin_mask])
                ece += np.abs(avg_confidence - avg_accuracy) * np.sum(bin_mask) / n
        return ece

    def get_clarity_score_distribution(self):
        """
        Computes and returns a distribution of clarity scores.
        
        Returns:
            dict: A dictionary with keys "average", "min", "max", and "count", or empty if no scores.
        """
        if not self.clarity_scores:
            return {}
        average = sum(self.clarity_scores) / len(self.clarity_scores)
        return {
            "average": average,
            "min": min(self.clarity_scores),
            "max": max(self.clarity_scores),
            "count": len(self.clarity_scores)
        }

    def get_ltm_hit_rate(self):
        """
        Computes the LTM Engagement Rate (percentage of samples where refinement was triggered).

        Returns:
            float: The LTM engagement rate as a percentage.
        """
        total = len(self.overall_ground_truth)
        return (self.ltm_engagement_count / total * 100) if total > 0 else 0.0