# Submit this file to Gradescope
from typing import Dict, List, Tuple
import sys
import math
# you may use other Python standard libraries, but not data
# science libraries, such as numpy, scikit-learn, etc.


class Solution:

  def confusion_matrix(self, true_labels: List[int], pred_labels: List[int]) -> Dict[Tuple[int, int], int]:
    """Calculate the confusion matrix and return it as a sparse matrix in dictionary form.
    Args:
      true_labels: list of true labels
      pred_labels: list of predicted labels
    Returns:
      A dictionary of (true_label, pred_label): count
    """
    counts: Dict[Tuple[int, int], int] = {}
    for t, p in zip(true_labels, pred_labels):
      key = (t, p)
      counts[key] = counts.get(key, 0) + 1
    return counts

  def jaccard(self, true_labels: List[int], pred_labels: List[int]) -> float:
    """Calculate the Jaccard index.
    Args:
      true_labels: list of true cluster labels
      pred_labels: list of predicted cluster labels
    Returns:
      The Jaccard index. Do NOT round this value.
    """
    n = len(true_labels)
    tp = 0
    fp = 0
    fn = 0
    for i in range(n):
      for j in range(i + 1, n):
        same_true = (true_labels[i] == true_labels[j])
        same_pred = (pred_labels[i] == pred_labels[j])
        if same_true and same_pred:
          tp += 1
        elif (not same_true) and same_pred:
          fp += 1
        elif same_true and (not same_pred):
          fn += 1
    denom = tp + fp + fn
    if denom == 0:
      return 0.0
    return tp / denom

  def nmi(self, true_labels: List[int], pred_labels: List[int]) -> float:
    """Calculate the normalized mutual information.
    Args:
      true_labels: list of true cluster labels
      pred_labels: list of predicted cluster labels
    Returns:
      The normalized mutual information. Do NOT round this value.
    """
    n = len(true_labels)
    if n == 0:
      return 0.0
    true_counts: Dict[int, int] = {}
    pred_counts: Dict[int, int] = {}
    joint_counts: Dict[Tuple[int, int], int] = {}
    for t, p in zip(true_labels, pred_labels):
      true_counts[t] = true_counts.get(t, 0) + 1
      pred_counts[p] = pred_counts.get(p, 0) + 1
      joint_counts[(t, p)] = joint_counts.get((t, p), 0) + 1

    def safe_log(x: float) -> float:
      return math.log(x) if x > 0.0 else 0.0

    H_true = 0.0
    for cnt in true_counts.values():
      p = cnt / n
      H_true -= p * safe_log(p)

    H_pred = 0.0
    for cnt in pred_counts.values():
      p = cnt / n
      H_pred -= p * safe_log(p)

    MI = 0.0
    for (t, p), cnt in joint_counts.items():
      p_xy = cnt / n
      p_x = true_counts[t] / n
      p_y = pred_counts[p] / n
      if p_xy > 0.0:
        MI += p_xy * (safe_log(p_xy) - safe_log(p_x * p_y))

    denom = H_true + H_pred
    if denom <= 0.0:
      return 0.0
    return 2.0 * MI / denom


def _read_pairs_from_stdin():
  data = sys.stdin.read().strip().splitlines()
  if not data:
    return None, None, None
  mode_line = data[0].strip()
  try:
    mode = int(mode_line)
  except Exception:
    mode = 0
  true = []
  pred = []
  for line in data[1:]:
    line = line.strip()
    if not line:
      continue
    parts = line.split()
    if len(parts) < 2:
      continue
    t = int(parts[0])
    p = int(parts[1])
    true.append(t)
    pred.append(p)
  return mode, true, pred


if __name__ == "__main__":
  mode, true_labels, pred_labels = _read_pairs_from_stdin()
  if true_labels is None:
    sys.exit(0)
  sol = Solution()
  if mode == 2:
    # confusion matrix: print nonzero entries sorted by true then pred
    cm = sol.confusion_matrix(true_labels, pred_labels)
    for (t, p) in sorted(cm.keys()):
      print(f"{t} {p} {cm[(t,p)]}")
  elif mode == 0:
    val = sol.jaccard(true_labels, pred_labels)
    print(f"{val:.4f}")
  else:
    val = sol.nmi(true_labels, pred_labels)
    print(f"{val:.4f}")
