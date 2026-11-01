import torch


@torch.no_grad()
def segmentation_metrics(logits, target, threshold=0.5):
    probs = torch.sigmoid(logits)

    if isinstance(threshold, (list, tuple)):
        threshold = torch.tensor(threshold, device=logits.device, dtype=probs.dtype).view(1, 3, 1, 1)

    preds = (probs >= threshold).float()

    dims = (0, 2, 3)

    tp = (preds * target).sum(dim=dims)
    fp = (preds * (1 - target)).sum(dim=dims)
    fn = ((1 - preds) * target).sum(dim=dims)

    return {'tp': tp, 'fp': fp, 'fn': fn}


def calculate_metrics(tp, fp, fn):

    def safe_divide(numerator, denominator):
        return torch.where(
            denominator > 0,
            numerator / denominator.clamp(min=1e-7),
            torch.full_like(denominator, float('nan'))
        )

    dice = safe_divide(2 * tp, 2 * tp + fp + fn)
    iou = safe_divide(tp, tp + fp + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    return {
    'dice': dice,
    'iou': iou,
    'precision': precision,
    'recall': recall,
    'mean_dice': torch.nanmean(dice).item(),
    'mean_iou': torch.nanmean(iou).item()
}
