import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from mmengine.evaluator import BaseMetric
from mmaction.registry import METRICS

@METRICS.register_module()
class PRF1AUCMetric(BaseMetric):
    

    def __init__(self, average='macro', collect_device='cpu', prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.average = average

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            self.results.append({
                'pred': sample['pred_score'].detach().cpu(),
                'label': sample['gt_label'].detach().cpu()
            })

    def compute_metrics(self, results):
        preds = torch.stack([r['pred'] for r in results]).numpy()
        labels = torch.cat([r['label'] for r in results]).numpy()
        pred_labels = preds.argmax(axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_labels, average=self.average, zero_division=0)
        try:
            roc = roc_auc_score(labels, preds, multi_class='ovr', average=self.average)
        except ValueError:
            roc = float('nan')
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc)
        }
