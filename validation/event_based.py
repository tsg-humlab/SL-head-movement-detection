import numpy as np

from models.hmm.failure_analysis import get_group_indices


def evaluate_events(labels, preds, allow_multi_hit=False, fps=24, tolerance=2):
    label_groups = get_group_indices(labels)
    label_groups_bg = get_group_indices(labels, target=1)
    preds_groups = get_group_indices(preds)

    tn = fp = 0
    tp = np.zeros(label_groups.shape[0])

    for pred_i, pred_group in enumerate(preds_groups):
        hit = False

        for label_i, label_group in enumerate(label_groups):
            overlap = range(
                max(pred_group[0], label_group[0] - fps * tolerance),
                min(pred_group[1], label_group[1] + fps * tolerance) + 1
            )

            if len(overlap) > 0:
                hit = True
                tp[label_i] = 1

                if not allow_multi_hit:
                    break

        if not hit:
            fp += 1

    tp = int(np.sum(tp))

    for label_i, label_group in enumerate(label_groups_bg):
        if (label_group[1] - label_group[0]) < ((fps * tolerance) / 2):
            continue

        clean = True

        for pred_i, pred_group in enumerate(preds_groups):
            overlap = range(
                max(pred_group[0], label_group[0] + fps * tolerance),
                min(pred_group[1], label_group[1] - fps * tolerance) + 1
            )

            if len(overlap) > 0:
                clean = False
                break

        if clean:
            tn += 1

    fn = label_groups.shape[0] - tp

    return np.array([[tn, fp],
                     [fn, tp]])


