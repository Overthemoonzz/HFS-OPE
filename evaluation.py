import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def reciprocal_rank_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the Mean Reciprocal Rank (MRR) score.

    Args:
        y_true (np.ndarray): A 1D array of ground-truth labels. These should be binary (0 or 1),
                                where 1 indicates the relevant item.
        y_pred (np.ndarray): A 1D array of predicted scores. These scores indicate the likelihood
                                of items being relevant.

    Returns:
        float: The mean reciprocal rank (MRR) score.

    Note:
        Both `y_true` and `y_pred` should be 1D arrays of the same length.
        The function assumes higher scores in `y_pred` indicate higher relevance.

    Examples:
        >>> y_true_1 = np.array([0, 0, 1])
        >>> y_pred_1 = np.array([0.5, 0.2, 0.1])
        >>> reciprocal_rank_score(y_true_1, y_pred_1)
            0.33

        >>> y_true_2 = np.array([0, 1, 1])
        >>> y_pred_2 = np.array([0.5, 0.2, 0.1])
        >>> reciprocal_rank_score(y_true_2, y_pred_2)
            0.5

        >>> y_true_3 = np.array([1, 1, 0])
        >>> y_pred_3 = np.array([0.5, 0.2, 0.1])
        >>> reciprocal_rank_score(y_true_3, y_pred_3)
            1.0

        >>> np.mean(
                [
                    reciprocal_rank_score(y_true, y_pred)
                    for y_true, y_pred in zip(
                        [y_true_1, y_true_2, y_true_3], [y_pred_1, y_pred_2, y_pred_3]
                    )
                ]
            )
            0.61
            mrr_score([y_true_1, y_true_2, y_true_3], [y_pred_1, y_pred_2, y_pred_3])
    """
    order = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, order)
    first_positive_rank = np.argmax(y_true) + 1
    return 1.0 / first_positive_rank


def dcg_score(y_true, y_pred, k):
    """
    Compute the Discounted Cumulative Gain (DCG) score at a particular rank `k`.

    Args:
        y_true (np.ndarray): A 1D or 2D array of ground-truth relevance labels.
                            Each element should be a non-negative integer.
        y_pred (np.ndarray): A 1D or 2D array of predicted scores. Each element is
                            a score corresponding to the predicted relevance.
        k (int, optional): The rank at which the DCG score is calculated. Defaults
                            to 10. If `k` is larger than the number of elements, it
                            will be truncated to the number of elements.

    Note:
        In case of a 2D array, each row represents a different sample.

    Returns:
        float: The calculated DCG score for the top `k` elements.

    Raises:
        ValueError: If `y_true` and `y_pred` have different shapes.

    Examples:
        >>> from sklearn.metrics import dcg_score as dcg_score_sklearn
        >>> y_true = np.array([1, 0, 0, 1, 0])
        >>> y_pred = np.array([0.5, 0.2, 0.1, 0.8, 0.4])
        >>> dcg_score(y_true, y_pred)
            1.6309297535714575
        >>> dcg_score_sklearn([y_true], [y_pred])
            1.6309297535714573
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """
    Compute the Normalized Discounted Cumulative Gain (NDCG) score at a rank `k`.

    Args:
        y_true (np.ndarray): A 1D or 2D array of ground-truth relevance labels.
                            Each element should be a non-negative integer. In case
                            of a 2D array, each row represents a different sample.
        y_pred (np.ndarray): A 1D or 2D array of predicted scores. Each element is
                            a score corresponding to the predicted relevance. The
                            array should have the same shape as `y_true`.
        k (int, optional): The rank at which the NDCG score is calculated. Defaults
                            to 10. If `k` is larger than the number of elements, it
                            will be truncated to the number of elements.

    Returns:
        float: The calculated NDCG score for the top `k` elements. The score ranges
                from 0 to 1, with 1 representing the perfect ranking.

    Examples:
        >>> from sklearn.metrics import ndcg_score as ndcg_score_sklearn
        >>> y_true = np.array([1, 0, 0, 1, 0])
        >>> y_pred = np.array([0.1, 0.2, 0.1, 0.8, 0.4])
        >>> ndcg_score([y_true], [y_pred])
            0.863780110436402
        >>> ndcg_score_sklearn([y_true], [y_pred])
            0.863780110436402
        >>>
    """
    best = dcg_score(y_true, y_true, k)
    if best == 0:
        return 0.0
    actual = dcg_score(y_true, y_pred, k)
    return actual / best


def mrr_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the Mean Reciprocal Rank (MRR) score.

    THIS MIGHT NOT ALL PROPER, TO BE DETERMIEND:
        - https://github.com/recommenders-team/recommenders/issues/2141

    Args:
        y_true (np.ndarray): A 1D array of ground-truth labels. These should be binary (0 or 1),
                                where 1 indicates the relevant item.
        y_pred (np.ndarray): A 1D array of predicted scores. These scores indicate the likelihood
                                of items being relevant.

    Returns:
        float: The mean reciprocal rank (MRR) score.

    Note:
        Both `y_true` and `y_pred` should be 1D arrays of the same length.
        The function assumes higher scores in `y_pred` indicate higher relevance.

    Examples:
        >>> y_true = np.array([[1, 0, 0, 1, 0]])
        >>> y_pred = np.array([[0.5, 0.2, 0.1, 0.8, 0.4]])
        >>> mrr_score(y_true, y_pred)
            0.75

    """
    order = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def auc_score_custom(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Area Under the Curve (AUC) score for the Receiver Operating Characteristic (ROC) curve using a
    custom method. This implementation is particularly useful for understanding basic ROC curve properties and
    for educational purposes to demonstrate how AUC scores can be manually calculated.

    This function may produce slightly different results compared to standard library implementations (e.g., sklearn's roc_auc_score)
    in cases where positive and negative predictions have the same score. The function treats the problem as a binary classification task,
    comparing the prediction scores for positive instances against those for negative instances directly.

    Args:
        y_true (np.ndarray): A binary array indicating the true classification (1 for positive class and 0 for negative class).
        y_pred (np.ndarray): An array of scores as predicted by a model, indicating the likelihood of each instance being positive.

    Returns:
        float: The calculated AUC score, representing the probability that a randomly chosen positive instance is ranked
                higher than a randomly chosen negative instance based on the prediction scores.

    Raises:
        ValueError: If `y_true` and `y_pred` do not have the same length or if they contain invalid data types.

    Examples:
        >>> y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        >>> y_pred = np.array([0.9999, 0.9838, 0.5747, 0.8485, 0.8624, 0.4502, 0.3357, 0.8985])
        >>> auc_score_custom(y_true, y_pred)
            0.9333333333333333
        >>> from sklearn.metrics import roc_auc_score
        >>> roc_auc_score(y_true, y_pred)
            0.9333333333333333

        An error will occur when pos/neg prediction have same score:
        >>> y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        >>> y_pred = np.array([0.9999, 0.8, 0.8, 0.8485, 0.8624, 0.4502, 0.3357, 0.8985])
        >>> auc_score_custom(y_true, y_pred)
            0.7333
        >>> roc_auc_score(y_true, y_pred)
            0.7667
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_true_bool = y_true.astype(np.bool_)
    # Index:
    pos_scores = y_pred[y_true_bool]
    neg_scores = y_pred[np.logical_not(y_true_bool)]
    # Arrange:
    pos_scores = np.repeat(pos_scores, len(neg_scores))
    neg_scores = np.tile(neg_scores, sum(y_true_bool))
    assert len(neg_scores) == len(pos_scores)
    return (pos_scores > neg_scores).sum() / len(neg_scores) if len(neg_scores) > 0 else -1

def hit_rate_at_k(y_true, y_pred, k=10):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    def single_hit(true_row, pred_row):
        order = np.argsort(pred_row)[::-1]
        topk = true_row[order[:k]]
        return float(np.any(topk))

    if y_true.ndim == 1:
        return single_hit(y_true, y_pred)
    # assume 2D: each row is one session
    hits = [
        single_hit(true_row, pred_row)
        for true_row, pred_row in zip(y_true, y_pred)
    ]
    return float(np.mean(hits))


