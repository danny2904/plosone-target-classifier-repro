
import numpy as np

def generate_report_string(model_name, cls_metrics, span_metrics, confusion_matrix=None):
    """
    Generates a formatted report string compatible with English baseline reporter style.
    """
    report = []
    report.append("=" * 80)
    report.append(f"RESULTS REPORT: {model_name}")
    report.append("=" * 80)
    report.append(f"{'Metric':<35} | {'Value':<10}")
    report.append("-" * 50)
    
    # GROUP 1: Classification & Performance
    report.append(">> GROUP 1: CLASSIFICATION PERFORMANCE")
    report.append(f"{'Cls Accuracy':<35} | {cls_metrics.get('Accuracy', 0):.4f}")
    
    if 'F1-Macro' in cls_metrics:
        report.append(f"{'Cls F1-Macro':<35} | {cls_metrics['F1-Macro']:.4f}")
    if 'Precision' in cls_metrics:
        report.append(f"{'Cls Precision (Macro)':<35} | {cls_metrics['Precision']:.4f}")
    if 'Recall' in cls_metrics:
        report.append(f"{'Cls Recall (Macro)':<35} | {cls_metrics['Recall']:.4f}")
        
    report.append("-" * 50)
    
    # GROUP 2: Hate Spans Detection
    report.append(">> GROUP 2: HATE SPANS DETECTION")
    
    # Table 3 Standard Metrics (ViHateT5 Paper)
    if 'Acc' in span_metrics:
        report.append(f"{'Acc':<35} | {span_metrics['Acc']:.4f}")
    if 'WF1' in span_metrics:
        report.append(f"{'WF1':<35} | {span_metrics['WF1']:.4f}")
    if 'MF1' in span_metrics:
        report.append(f"{'MF1':<35} | {span_metrics['MF1']:.4f}")
    
    # Legacy Token Level (if present)
    if 'Token Accuracy' in span_metrics:
        report.append(f"{'Token Accuracy':<35} | {span_metrics['Token Accuracy']:.4f}")
    if 'Token mF1' in span_metrics: # Macro F1
        report.append(f"{'Token mF1':<35} | {span_metrics['Token mF1']:.4f}")
    if 'Token wF1' in span_metrics: # Weighted F1
        report.append(f"{'Token wF1':<35} | {span_metrics['Token wF1']:.4f}")
        
    # Syllable Level (VN Specific)
    if 'Syllable Accuracy' in span_metrics:
        report.append(f"{'Syllable Accuracy':<35} | {span_metrics['Syllable Accuracy']:.4f}")
    if 'Syllable F1 (Macro)' in span_metrics:
        report.append(f"{'Syllable F1 (Macro)':<35} | {span_metrics['Syllable F1 (Macro)']:.4f}")
    if 'Syllable F1 (Binary)' in span_metrics:
        report.append(f"{'Syllable F1 (Binary)':<35} | {span_metrics['Syllable F1 (Binary)']:.4f}")
        
    # Strict / IOU
    if 'Span IoU' in span_metrics:
        report.append(f"{'Span IoU':<35} | {span_metrics['Span IoU']:.4f}")
    if 'Char F1 (Macro)' in span_metrics:
        report.append(f"{'Char F1 (Macro)':<35} | {span_metrics['Char F1 (Macro)']:.4f}")
    if 'Char F1 (Binary)' in span_metrics:
        report.append(f"{'Char F1 (Binary)':<35} | {span_metrics['Char F1 (Binary)']:.4f}")
        
    report.append("-" * 50)
    
    if confusion_matrix is not None:
        report.append("CONFUSION MATRIX:")
        report.append(np.array2string(confusion_matrix, separator=', '))
    
    report.append("=" * 80)
    return "\n".join(report)
