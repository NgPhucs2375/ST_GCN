#!/usr/bin/env python3
"""
Confusion Matrix Analyzer - Phân tích tự động để cải thiện weak classes
Usage: python analyze_confusion_matrix.py --cm-file outputs_kaggle/confusion_matrix.pt
"""

import json
import numpy as np
import torch
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class ConfusionMatrixAnalyzer:
    """Analyze confusion matrix to identify weak classes and improvement strategies"""
    
    def __init__(self, confusion_matrix, class_names):
        """
        Args:
            confusion_matrix: (n_classes, n_classes) numpy array
            class_names: list of class name strings
        """
        self.cm = np.array(confusion_matrix, dtype=np.float32)
        self.classes = class_names
        self.n_classes = len(class_names)
        
        # Calculate metrics
        self._compute_metrics()
    
    def _compute_metrics(self):
        """Compute recall, precision, F1, support for each class"""
        self.recalls = np.diag(self.cm) / (self.cm.sum(axis=1) + 1e-10)
        self.precisions = np.diag(self.cm) / (self.cm.sum(axis=0) + 1e-10)
        
        # F1 score
        self.f1_scores = 2 * (self.precisions * self.recalls) / (
            self.precisions + self.recalls + 1e-10
        )
        
        # Support (number of samples)
        self.supports = self.cm.sum(axis=1)
        
        # Overall accuracy
        self.overall_accuracy = np.trace(self.cm) / self.cm.sum()
    
    def print_header(self):
        """Print analysis header"""
        print("\n" + "="*80)
        print("CONFUSION MATRIX ANALYSIS REPORT")
        print("="*80)
        print(f"Overall Accuracy: {self.overall_accuracy:.2%}")
        print(f"Total Samples: {int(self.cm.sum())}")
        print(f"Total Classes: {self.n_classes}")
        print("="*80)
    
    def show_weak_classes(self, top_k=5):
        """Show weakest classes sorted by recall (accuracy)"""
        print(f"\n📉 WEAK CLASSES (Top {top_k} by Recall):")
        print("-" * 80)
        
        # Sort by recall
        sorted_idx = np.argsort(self.recalls)
        
        print(f"{'Rank':<5} {'Class':<8} {'Recall':<10} {'Precision':<12} "
              f"{'F1':<10} {'n':<6}")
        print("-" * 80)
        
        for rank, idx in enumerate(sorted_idx[:top_k], 1):
            cls = self.classes[idx]
            recall = self.recalls[idx]
            precision = self.precisions[idx]
            f1 = self.f1_scores[idx]
            support = int(self.supports[idx])
            
            # Color code by performance
            if recall < 0.75:
                status = "🔴 CRITICAL"
            elif recall < 0.85:
                status = "🟡 WEAK"
            else:
                status = "🟢 OK"
            
            print(f"{rank:<5} {cls:<8} {recall:>8.1%}  {precision:>10.1%}  "
                  f"{f1:>8.1%}  {support:>4}  {status}")
        
        print("-" * 80)
    
    def show_strong_classes(self, top_k=3):
        """Show strongest classes sorted by recall"""
        print(f"\n✅ STRONG CLASSES (Top {top_k} by Recall):")
        print("-" * 80)
        
        # Sort by recall (descending)
        sorted_idx = np.argsort(self.recalls)[::-1][:top_k]
        
        print(f"{'Rank':<5} {'Class':<8} {'Recall':<10} {'Precision':<12} "
              f"{'F1':<10} {'n':<6}")
        print("-" * 80)
        
        for rank, idx in enumerate(sorted_idx, 1):
            cls = self.classes[idx]
            recall = self.recalls[idx]
            precision = self.precisions[idx]
            f1 = self.f1_scores[idx]
            support = int(self.supports[idx])
            
            print(f"{rank:<5} {cls:<8} {recall:>8.1%}  {precision:>10.1%}  "
                  f"{f1:>8.1%}  {support:>4}")
        
        print("-" * 80)
    
    def show_confusion_patterns(self):
        """Show what each weak class gets confused with"""
        print("\n🔀 CONFUSION PATTERNS - Where weak classes go wrong:")
        print("-" * 80)
        
        # Analyze top 5 weakest classes
        weak_idx = np.argsort(self.recalls)[:5]
        
        for class_idx in weak_idx:
            cls_name = self.classes[class_idx]
            recall = self.recalls[class_idx]
            
            # Get confusions for this class (exclude diagonal)
            confusions = self.cm[class_idx, :].copy()
            correct = confusions[class_idx]
            confusions[class_idx] = 0  # Exclude correct predictions
            
            if confusions.sum() == 0:
                continue  # No confusions for this class
            
            print(f"\n{cls_name} (Recall: {recall:.1%}, n={int(self.supports[class_idx])}):")
            print(f"  ✓ Correct: {int(correct)}")
            
            # Top 3 confusions
            top_confused_idx = np.argsort(confusions)[-3:][::-1]
            
            for rank, conf_idx in enumerate(top_confused_idx, 1):
                if confusions[conf_idx] > 0:
                    count = int(confusions[conf_idx])
                    pct = confusions[conf_idx] / self.cm[class_idx, :].sum() * 100
                    confused_cls = self.classes[conf_idx]
                    
                    # Emoji for severity
                    if pct > 15:
                        emoji = "🔴"
                    elif pct > 8:
                        emoji = "🟡"
                    else:
                        emoji = "🟢"
                    
                    print(f"  {emoji} → {confused_cls}: {count} samples ({pct:.1f}%)")
        
        print("-" * 80)
    
    def show_pattern_analysis(self):
        """Analyze Precision vs Recall patterns for improvement guidance"""
        print("\n💡 PATTERN ANALYSIS - What to improve:")
        print("-" * 80)
        
        weak_idx = np.argsort(self.recalls)[:5]
        
        for class_idx in weak_idx:
            cls = self.classes[class_idx]
            recall = self.recalls[class_idx]
            precision = self.precisions[class_idx]
            support = int(self.supports[class_idx])
            
            print(f"\n{cls} (n={support}):")
            print(f"  Recall:    {recall:.1%} {'⬇️ LOW' if recall < 0.80 else '✓'}")
            print(f"  Precision: {precision:.1%} {'⬇️ LOW' if precision < 0.85 else '✓'}")
            
            # Determine pattern
            if recall < 0.80 and precision > 0.85:
                pattern = "HIGH PRECISION, LOW RECALL"
                cause = "Model is conservative (predicts class rarely)"
                solutions = [
                    "✓ Already applied: WeightedSampler (oversample)",
                    "✓ Already applied: Class-weighted loss",
                    "✓ Already applied: Label smoothing (0.15)",
                    "→ Next: Try threshold adjustment or more augmentation"
                ]
            elif recall < 0.80 and precision < 0.85:
                pattern = "LOW RECALL, LOW PRECISION"
                cause = "Model struggles with this class (hard to learn)"
                solutions = [
                    "→ Check data quality (mislabeled?)",
                    "→ Add more augmentation",
                    "→ Increase model capacity",
                    "→ Longer training or different scheduler"
                ]
            elif recall > 0.80 and precision < 0.85:
                pattern = "HIGH RECALL, LOW PRECISION"
                cause = "Model predicts this class too often (overconfident)"
                solutions = [
                    "→ Reduce class weight",
                    "→ Increase dropout/regularization",
                    "→ Check for mislabeled data"
                ]
            else:
                pattern = "BALANCED"
                cause = "Doing well, minor room for improvement"
                solutions = []
            
            print(f"  Pattern: {pattern}")
            print(f"  Cause: {cause}")
            if solutions:
                print(f"  Solutions:")
                for sol in solutions:
                    print(f"    {sol}")
        
        print("-" * 80)
    
    def show_misclassification_summary(self):
        """Show total misclassification breakdown"""
        print("\n📊 MISCLASSIFICATION SUMMARY:")
        print("-" * 80)
        
        total_samples = int(self.cm.sum())
        correct = int(np.trace(self.cm))
        incorrect = total_samples - correct
        
        print(f"Total samples: {total_samples}")
        print(f"Correct predictions: {correct} ({correct/total_samples:.1%})")
        print(f"Incorrect predictions: {incorrect} ({incorrect/total_samples:.1%})")
        
        # Per-class error
        print(f"\nErrors per class (FN + FP):")
        print(f"{'Class':<8} {'Errors':<8} {'Error Rate':<12} {'Type':<30}")
        print("-" * 80)
        
        for idx in np.argsort(self.recalls)[:5]:  # Top 5 weakest
            cls = self.classes[idx]
            
            # False negatives (actual class but predicted as something else)
            fn = self.cm[idx, :].sum() - self.cm[idx, idx]
            
            # False positives (predicted as this class but actually something else)
            fp = self.cm[:, idx].sum() - self.cm[idx, idx]
            
            total_errors = fn + fp
            error_rate = total_errors / self.cm[idx, :].sum() if self.cm[idx, :].sum() > 0 else 0
            
            error_type = f"FN:{int(fn)}, FP:{int(fp)}"
            
            print(f"{cls:<8} {int(total_errors):<8} {error_rate:>10.1%}  {error_type:<30}")
        
        print("-" * 80)
    
    def save_heatmap(self, output_path="confusion_matrix_heatmap.png"):
        """Save confusion matrix heatmap visualization"""
        try:
            # Normalize for better visualization
            cm_norm = self.cm / self.cm.sum(axis=1, keepdims=True)
            
            plt.figure(figsize=(14, 12))
            sns.heatmap(cm_norm, annot=self.cm.astype(int), fmt='d',
                       xticklabels=self.classes, yticklabels=self.classes,
                       cmap='Blues', cbar_kws={'label': 'Proportion'})
            
            plt.title('Confusion Matrix - Gesture Recognition', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Class', fontsize=12)
            plt.ylabel('Actual Class', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✅ Heatmap saved to: {output_path}")
            plt.close()
        except Exception as e:
            print(f"\n⚠️ Could not save heatmap: {e}")
    
    def save_metrics_csv(self, output_path="confusion_matrix_metrics.csv"):
        """Save per-class metrics to CSV"""
        try:
            import csv
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Class', 'Recall', 'Precision', 'F1', 'Support'])
                
                for idx in range(self.n_classes):
                    writer.writerow([
                        self.classes[idx],
                        f"{self.recalls[idx]:.4f}",
                        f"{self.precisions[idx]:.4f}",
                        f"{self.f1_scores[idx]:.4f}",
                        int(self.supports[idx])
                    ])
            
            print(f"✅ Metrics saved to: {output_path}")
        except Exception as e:
            print(f"⚠️ Could not save CSV: {e}")
    
    def generate_full_report(self, output_dir="."):
        """Generate complete analysis report"""
        self.print_header()
        self.show_weak_classes(top_k=5)
        self.show_strong_classes(top_k=3)
        self.show_confusion_patterns()
        self.show_pattern_analysis()
        self.show_misclassification_summary()
        
        # Save visualizations
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        self.save_heatmap(str(output_dir / "confusion_matrix_heatmap.png"))
        self.save_metrics_csv(str(output_dir / "confusion_matrix_metrics.csv"))
        
        print("\n" + "="*80)
        print("✅ Analysis complete! Check outputs:")
        print(f"  - confusion_matrix_heatmap.png")
        print(f"  - confusion_matrix_metrics.csv")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze confusion matrix to identify weak classes'
    )
    parser.add_argument('--cm-file', type=str, 
                       default='outputs_kaggle/confusion_matrix.pt',
                       help='Path to confusion_matrix.pt file')
    parser.add_argument('--labels-file', type=str,
                       default='outputs_kaggle/labels.json',
                       help='Path to labels.json file')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Load confusion matrix
    try:
        print(f"📂 Loading confusion matrix from: {args.cm_file}")
        cm = torch.load(args.cm_file).numpy()
    except FileNotFoundError:
        print(f"❌ File not found: {args.cm_file}")
        return
    
    # Load class labels
    try:
        print(f"📂 Loading class labels from: {args.labels_file}")
        with open(args.labels_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'class_names' in data:
                class_names = data['class_names']
            else:
                class_names = data if isinstance(data, list) else list(data.values())
    except FileNotFoundError:
        print(f"❌ File not found: {args.labels_file}")
        return
    
    # Analyze
    analyzer = ConfusionMatrixAnalyzer(cm, class_names)
    analyzer.generate_full_report(args.output_dir)


if __name__ == '__main__':
    main()
