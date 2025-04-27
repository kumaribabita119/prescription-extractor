

import numpy as np
from Levenshtein import distance as levenshtein_distance
import json
import matplotlib.pyplot as plt

class PrescriptionEvaluator:
    def __init__(self):
        """Initialize the prescription evaluator"""
        self.field_weights = {
            "patient_name": 1.0,
            "patient_age": 0.8,
            "patient_gender": 0.8,
            "doctor_name": 1.0,
            "date": 0.9,
            "diagnosis": 0.9,
            "hospital/clinic": 0.7
        }
        
        # Weights for medication fields
        self.medication_weights = {
            "name": 1.0,
            "dosage": 0.9,
            "frequency": 0.9,
            "duration": 0.8,
            "route": 0.7,
            "special_instructions": 0.6
        }
        
    def load_data(self, predictions_path, ground_truth_path):
        """
        Load prediction and ground truth data
        
        Args:
            predictions_path: Path to predictions JSON file
            ground_truth_path: Path to ground truth JSON file
            
        Returns:
            Tuple of (predictions, ground_truth)
        """
        with open(predictions_path, 'r') as f:
            predictions = json.load(f)
        
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)
            
        return predictions, ground_truth
    
    def calculate_string_similarity(self, str1, str2):
        """
        Calculate similarity between two strings using Levenshtein distance
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score (0-1)
        """
        if str1 is None and str2 is None:
            return 1.0
        if str1 is None or str2 is None:
            return 0.0
            
        str1 = str(str1).lower().strip()
        str2 = str(str2).lower().strip()
        
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
            
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
            
        lev_dist = levenshtein_distance(str1, str2)
        similarity = 1 - (lev_dist / max_len)
        
        return max(0, similarity)
    
    def evaluate_single_prescription(self, prediction, ground_truth):
        """
        Evaluate extraction performance for a single prescription
        
        Args:
            prediction: Predicted prescription data
            ground_truth: Ground truth prescription data
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            "field_scores": {},
            "medication_scores": [],
            "overall_score": 0.0
        }
        
        # Evaluate simple fields
        total_weight = 0
        weighted_score = 0
        
        for field, weight in self.field_weights.items():
            if field in ground_truth:
                total_weight += weight
                
                if field in prediction:
                    similarity = self.calculate_string_similarity(prediction[field], ground_truth[field])
                    metrics["field_scores"][field] = similarity
                    weighted_score += similarity * weight
                else:
                    metrics["field_scores"][field] = 0.0
        
        # Evaluate medications
        if "medication_list" in ground_truth and "medication_list" in prediction:
            gt_meds = ground_truth["medication_list"]
            pred_meds = prediction["medication_list"]
            
            # Try to match medications by name
            for gt_med in gt_meds:
                best_match = None
                best_score = 0
                
                for pred_med in pred_meds:
                    if "name" in gt_med and "name" in pred_med:
                        name_sim = self.calculate_string_similarity(pred_med["name"], gt_med["name"])
                        
                        if name_sim > 0.7 and name_sim > best_score:  # Match if name is similar enough
                            best_match = pred_med
                            best_score = name_sim
                
                # Evaluate matched medication
                med_metrics = {"name": best_score if best_match else 0.0}
                med_weight = self.medication_weights["name"]
                med_score = best_score * med_weight if best_match else 0
                
                # Evaluate other medication fields
                if best_match:
                    for field, weight in self.medication_weights.items():
                        if field != "name" and field in gt_med:
                            med_weight += weight
                            
                            if field in best_match:
                                similarity = self.calculate_string_similarity(best_match[field], gt_med[field])
                                med_metrics[field] = similarity
                                med_score += similarity * weight
                            else:
                                med_metrics[field] = 0.0
                
                metrics["medication_scores"].append({
                    "ground_truth": gt_med.get("name", "Unknown"),
                    "predicted": best_match.get("name", "Not found") if best_match else "Not found",
                    "field_scores": med_metrics,
                    "score": med_score / med_weight if med_weight > 0 else 0
                })
                
                # Add to overall weighted score
                weighted_score += med_score
                total_weight += med_weight
        
        # Calculate overall score
        metrics["overall_score"] = weighted_score / total_weight if total_weight > 0 else 0
        
        return metrics
    
    def evaluate_dataset(self, predictions, ground_truth):
        """
        Evaluate extraction performance for the entire dataset
        
        Args:
            predictions: List of predicted prescription data
            ground_truth: List of ground truth prescription data
            
        Returns:
            Dictionary of evaluation metrics
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Number of predictions and ground truth samples must match")
            
        all_metrics = []
        for i in range(len(predictions)):
            metrics = self.evaluate_single_prescription(predictions[i], ground_truth[i])
            all_metrics.append(metrics)
        
        # Calculate average metrics
        avg_metrics = {
            "overall_score": np.mean([m["overall_score"] for m in all_metrics]),
            "field_scores": {},
            "medication_field_scores": {}
        }
        
        # Average field scores
        all_fields = set()
        for metrics in all_metrics:
            all_fields.update(metrics["field_scores"].keys())
            
        for field in all_fields:
            scores = [m["field_scores"].get(field, 0) for m in all_metrics if field in m["field_scores"]]
            if scores:
                avg_metrics["field_scores"][field] = np.mean(scores)
        
        # Average medication field scores
        med_fields = set()
        for metrics in all_metrics:
            for med_score in metrics["medication_scores"]:
                med_fields.update(med_score["field_scores"].keys())
                
        for field in med_fields:
            all_scores = []
            for metrics in all_metrics:
                for med_score in metrics["medication_scores"]:
                    if field in med_score["field_scores"]:
                        all_scores.append(med_score["field_scores"][field])
            
            if all_scores:
                avg_metrics["medication_field_scores"][field] = np.mean(all_scores)
        
        return avg_metrics
    
    def visualize_results(self, metrics, output_path=None):
        """
        Create visualizations of evaluation results
        
        Args:
            metrics: Evaluation metrics
            output_path: Path to save visualization
        """
        # Create bar chart of field scores
        plt.figure(figsize=(12, 6))
        
        # Plot simple fields
        fields = list(metrics["field_scores"].keys())
        scores = list(metrics["field_scores"].values())
        
        x = np.arange(len(fields))
        plt.bar(x, scores, width=0.4, label='Patient & Doctor Fields')
        
        # Plot medication fields
        med_fields = list(metrics["medication_field_scores"].keys())
        med_scores = list(metrics["medication_field_scores"].values())
        
        x2 = np.arange(len(fields), len(fields) + len(med_fields))
        plt.bar(x2, med_scores, width=0.4, label='Medication Fields', color='orange')
        
        # Add labels and title
        plt.xlabel('Fields')
        plt.ylabel('Average Similarity Score (0-1)')
        plt.title('Prescription Extraction Performance by Field')
        plt.xticks(np.concatenate([x, x2]), fields + med_fields, rotation=45, ha='right')
        plt.ylim(0, 1.1)
        
        # Add overall score as text
        plt.text(0.5, 1.05, f'Overall Score: {metrics["overall_score"]:.2f}', 
                 horizontalalignment='center', transform=plt.gca().transAxes, fontsize=12)
        
        plt.legend()
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()
            