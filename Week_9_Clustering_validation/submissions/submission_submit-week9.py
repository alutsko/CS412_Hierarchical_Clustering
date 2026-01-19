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
    Counts: Dict[Tuple[int, int], int] = {}
    true_counts: Dict[int, int] = {}
    pred_counts: Dict[int, int] = {}
    joint_counts: Dict[Tuple[int, int], int] = {}
    for t, p in zip(true_labels, pred_labels):
      key = (t, p)
      Counts[key] = Counts.get(key, 0) + 1
      true_counts[t] = true_counts.get(t, 0) + 1
      pred_counts[p] = pred_counts.get(p, 0) + 1
      joint_counts[(t, p)] = joint_counts.get((t, p), 0) + 1

    #print("nonzero entries =", len(Counts), " sample:", list(Counts.items())[:5])

    # def safe_log_cm(x: float) -> float:
    #   return math.log(x) if x > 0.0 else 0.0
    #
    # HTrue_cm = 0.0
    # for cnt in true_counts.values():
    #   p = cnt / max(1, len(true_labels))
    #   HTrue_cm -= p * safe_log_cm(p)
    #
    # HPred_cm = 0.0
    # for cnt in pred_counts.values():
    #   p = cnt / max(1, len(true_labels))
    #   HPred_cm -= p * safe_log_cm(p)
    #
    # MutualInfo_cm = 0.0
    # for (t, p), cnt in joint_counts.items():
    #   p_xy = cnt / max(1, len(true_labels))
    #   p_x = true_counts[t] / max(1, len(true_labels))
    #   p_y = pred_counts[p] / max(1, len(true_labels))
    #   if p_xy > 0.0:
    #     MutualInfo_cm += p_xy * (safe_log_cm(p_xy) - safe_log_cm(p_x * p_y))
    #
    # denom_cm = HTrue_cm + HPred_cm

    return Counts

  def jaccard(self, true_labels: List[int], pred_labels: List[int]) -> float:
    """Calculate the Jaccard index.
    Args:
      true_labels: list of true cluster labels
      pred_labels: list of predicted cluster labels
    Returns:
      The Jaccard index. Do NOT round this value.
    """
    n = len(true_labels)
    Tp = 0
    Fp = 0
    Fn = 0

    for i in range(n):
      for j in range(i + 1, n):
        same_true = (true_labels[i] == true_labels[j])
        same_pred = (pred_labels[i] == pred_labels[j])
        if same_true and same_pred:
          Tp += 1
        elif (not same_true) and same_pred:
          Fp += 1
        elif same_true and (not same_pred):
          Fn += 1
    denom = Tp + Fp + Fn

  # print(" Tp:", Tp, " Fp:", Fp, " Fn:", Fn, " denom:", denom)

    if denom == 0:
      result = 0.0
      return result
    result = Tp / denom
    return result

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
      result = 0.0
      return result
    
    true_counts: Dict[int, int] = {}
    pred_counts: Dict[int, int] = {}
    joint_counts: Dict[Tuple[int, int], int] = {}

    for t, p in zip(true_labels, pred_labels):
      true_counts[t] = true_counts.get(t, 0) + 1
      pred_counts[p] = pred_counts.get(p, 0) + 1
      joint_counts[(t, p)] = joint_counts.get((t, p), 0) + 1

    def safe_log(x: float) -> float:
      return math.log(x) if x > 0.0 else 0.0

  # print("n:", n, " true_clusters:", len(true_counts), " pred_clusters:", len(pred_counts), " joint_entries:", len(joint_counts))

    HTrue = 0.0
    for cnt in true_counts.values():
      p = cnt / n
      HTrue -= p * safe_log(p)

    HPred = 0.0
    for cnt in pred_counts.values():
      p = cnt / n
      HPred -= p * safe_log(p)

    MutualInfo = 0.0
    for (t, p), cnt in joint_counts.items():
      p_xy = cnt / n
      p_x = true_counts[t] / n
      p_y = pred_counts[p] / n
      if p_xy > 0.0:
        MutualInfo += p_xy * (safe_log(p_xy) - safe_log(p_x * p_y))

    denom = HTrue + HPred

    if denom <= 0.0:
      result = 0.0
      return result
    result = 2.0 * MutualInfo / denom
    return result


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

    cm = sol.confusion_matrix(true_labels, pred_labels)
    for (t, p) in sorted(cm.keys()):
      print(f"{t} {p} {cm[(t,p)]}")
  elif mode == 0:
    val = sol.jaccard(true_labels, pred_labels)
    print(f"{val:.4f}")
  else:
    val = sol.nmi(true_labels, pred_labels)
    print(f"{val:.4f}")
