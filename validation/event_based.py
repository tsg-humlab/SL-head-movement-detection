import numpy as np

from models.processing.failure_analysis import get_group_indices


def evaluate_events(labels, predictions, allow_multi_hit=True, fps=25, tolerance=2):
    label_groups_shake, label_groups_nod = get_group_indices(labels, shake=True, nod=True)
    label_groups_bg = get_group_indices(labels, target=1, shake=False, nod=False)
    predictions_groups_shake, predictions_groups_nod = get_group_indices(predictions, shake=True, nod=True)
    predictions_groups_bg = get_group_indices(predictions, target=1, shake=False, nod=False)

    label_groups_array = [label_groups_shake, label_groups_nod]
    predictions_groups_array = [predictions_groups_shake, predictions_groups_nod]
    # label_groups_bg_array = [label_groups_bg_shake, label_groups_bg_nod]
    evals = []

    # First do shake, then nod
    for i, _ in enumerate(label_groups_array):
        label_groups = np.array(label_groups_array[i])
        predictions_groups = np.array(predictions_groups_array[i])
        # label_groups_bg = np.array(label_groups_bg_array[i])

        tn = fp = 0
        tp = np.zeros(label_groups.shape[0])

        for _, pred_group in enumerate(predictions_groups):
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

            for _, pred_group in enumerate(predictions_groups):
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

        evals.append(np.array([[tn, fp],
                               [fn, tp]]))

    return evals

