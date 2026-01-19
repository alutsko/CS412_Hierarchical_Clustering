from typing import List
import sys
import math

class Node:

  """
  This class, Node, represents a single node in a decision tree. It is designed to store information about the tree
  structure and the specific split criteria at each node. It is important to note that this class should NOT be
  modified as it is part of the assignment and will be used by the autograder.

  The attributes of the Node class are:
  - split_dim: The dimension/feature along which the node splits the data (-1 by default, indicating uninitialized)
  - split_point: The value used for splitting the data at this node (-1 by default, indicating uninitialized)
  - label: The class label assigned to this node, which is the majority label of the data at this node. If there is a tie,
    the numerically smaller label is assigned (-1 by default, indicating uninitialized)
  - left: The left child node of this node (None by default). Either None or a Node object.
  - right: The right child node of this node (None by default) Either None or a Node object.
  """

  def __init__(self):
    self.split_dim = -1
    self.split_point = -1
    self.label = -1
    self.left = None
    self.right = None


class Solution:
  """
  Example usage of the Node class to build a decision tree using a custom method called split_node():

  # In the fit method, create the root node and call the split_node() method to build the decision tree
    self.root = Node()
    self.split_node(self.root, data, ..., depth=0)

  def split_node(self, node, data, ..., depth):
      # Your implementation to calculate split_dim, split_point, and label for the given node and data
      # ...

      # Assign the calculated values to the node
      node.split_dim = split_dim
      node.split_point = split_point
      node.label = label

      # Recursively call split_node() for the left and right child nodes if the current node is not a leaf node
      # Remember, a leaf node is one that either only has data from one class or one that is at the maximum depth
      if not is_leaf:
          left_child = Node()
          right_child = Node()

          split_node(left_child, left_data, ..., depth+1)
          split_node(right_child, right_data, ..., depth+1)
  """

  def split_info(self, data: List[List[float]], label: List[int], split_dim: int, split_point: float) -> float:
    """
    Compute the information needed to classify a dataset if it's split
    with the given splitting dimension and splitting point, i.e. Info_A in the slides.

    Parameters:
    data (List[List]): A nested list representing the dataset.
    label (List): A list containing the class labels for each data point.
    split_dim (int): The dimension/attribute index to split the data on.
    split_point (float): The value at which the data should be split along the given dimension.

    Returns:
    float: The calculated Info_A value for the given split. Do NOT round this value
    """
    # Build left and right partitions based on split
    n = len(data)
    if n == 0:
      return 0.0

    left_counts = {}
    right_counts = {}
    nL = 0
    nR = 0
    for x, y in zip(data, label):
      v = x[split_dim]
      if v <= split_point:
        left_counts[y] = left_counts.get(y, 0) + 1
        nL += 1
      else:
        right_counts[y] = right_counts.get(y, 0) + 1
        nR += 1

    def entropy(counts, total):
      if total == 0:
        return 0.0
      H = 0.0
      for cnt in counts.values():
        p = cnt / total
        if p > 0.0:
          H -= p * math.log2(p)
      return H

    HL = entropy(left_counts, nL)
    HR = entropy(right_counts, nR)

    InfoA = 0.0
    if n > 0:
      InfoA = (nL / n) * HL + (nR / n) * HR
    return InfoA

  def fit(self, train_data: List[List[float]], train_label: List[int]) -> None:

    self.root = Node()

    """
    Fit the decision tree model using the provided training data and labels.

    Parameters:
    train_data (List[List[float]]): A nested list of floating point numbers representing the training data.
    train_label (List[int]): A list of integers representing the class labels for each data point in the training set.

    This method initializes the decision tree model by creating the root node. It then builds the decision tree starting 
    from the root node
    
    It is important to note that for tree structure evaluation, the autograder for this assignment
    first calls this method. It then performs tree traversals starting from the root node in order to check whether 
    the tree structure is correct. 
    
    So it is very important to ensure that self.root is assigned correctly to the root node
    
    It is best to use a different method (such as in the example above) to build the decision tree.
    """
    # Simple recursive tree builder using midpoints between unique values as split candidates.
    if not train_data or not train_label:
      self.root = Node()
      return

    n = len(train_data)
    D = 0
    for x in train_data:
      if len(x) > D:
        D = len(x)

    def majority_label(labels):
      counts = {}
      for l in labels:
        counts[l] = counts.get(l, 0) + 1
      best = None
      bestc = -1
      for k, c in counts.items():
        if c > bestc or (c == bestc and (best is None or k < best)):
          best = k
          bestc = c
      return best

    def candidate_splits(values):
      # return list of (value, is_observed) candidates: observed unique values and midpoints
      uniq = sorted(set(values))
      cands = []
      for u in uniq:
        cands.append((u, True))
      for i in range(len(uniq) - 1):
        mid = (uniq[i] + uniq[i+1]) / 2.0
        cands.append((mid, False))
      # sort by value to make deterministic
      cands.sort(key=lambda x: x[0])
      return cands

    def info_for_split(indices, d, sp):
      left = [train_label[i] for i in indices if train_data[i][d] <= sp]
      right = [train_label[i] for i in indices if train_data[i][d] > sp]
      def ent(ls):
        if not ls:
          return 0.0
        cnts = {}
        for v in ls:
          cnts[v] = cnts.get(v, 0) + 1
        total = len(ls)
        H = 0.0
        for cnt in cnts.values():
          p = cnt / total
          if p > 0.0:
            H -= p * math.log2(p)
        return H
      nL = len(left)
      nR = len(right)
      if nL + nR == 0:
        return float('inf')
      return (nL / (nL + nR)) * ent(left) + (nR / (nL + nR)) * ent(right)

    def best_split(indices):
      best_d = -1
      best_sp = -1.0
      best_info = float('inf')
      for d in range(D):
        vals = [train_data[i][d] for i in indices]
        cand = candidate_splits(vals)
        for sp, is_obs in cand:
          inf = info_for_split(indices, d, sp)
          choose = False
          if inf < best_info - 1e-12:
            choose = True
          elif abs(inf - best_info) <= 1e-12:
            if d < best_d:
              choose = True
            elif d == best_d:
              # prefer observed value over midpoint
              if is_obs and not getattr(sys.modules[__name__], '___best_is_obs', False):
                choose = True
              elif (is_obs == getattr(sys.modules[__name__], '___best_is_obs', False)):
                if sp < best_sp:
                  choose = True
          if choose:
            best_info = inf
            best_d = d
            best_sp = sp
            # store whether best is observed in module-level temp (small hack to avoid extra variable capture)
            setattr(sys.modules[__name__], '___best_is_obs', is_obs)
      return best_d, best_sp, best_info

    def build(node, indices):
      node.label = majority_label([train_label[i] for i in indices])
      if len(set([train_label[i] for i in indices])) == 1:
        node.split_dim = -1
        node.split_point = -1
        node.left = None
        node.right = None
        return
      bd, bp, bi = best_split(indices)
      if bd == -1:
        node.split_dim = -1
        node.split_point = -1
        node.left = None
        node.right = None
        return
      node.split_dim = bd
      node.split_point = bp
      left_idx = [i for i in indices if train_data[i][bd] <= bp]
      right_idx = [i for i in indices if train_data[i][bd] > bp]
      if not left_idx or not right_idx:
        node.split_dim = -1
        node.split_point = -1
        node.left = None
        node.right = None
        return
      node.left = Node()
      node.right = Node()
      build(node.left, left_idx)
      build(node.right, right_idx)

    all_idx = list(range(len(train_data)))
    build(self.root, all_idx)

  def classify(self, train_data: List[List[float]], train_label: List[int], test_data: List[List[float]]) -> List[int]:
    """
    Classify the test data using a decision tree model built from the provided training data and labels.
    This method first fits the decision tree model using the provided training data and labels by calling the
    'fit()' method.

    Parameters:
    train_data (List[List[float]]): A nested list of floating point numbers representing the training data.
    train_label (List[int]): A list of integers representing the class labels for each data point in the training set.
    test_data (List[List[float]]): A nested list of floating point numbers representing the test data.

    Returns:
    List[int]: A list of integer predictions, which are the label predictions for the test data after fitting
               the train data and labels to a decision tree.
    """
    # fit the tree first
    self.fit(train_data, train_label)
    # helper to traverse
    def predict_one(x):
      node = self.root
      while node is not None:
        if node.split_dim == -1 or node.left is None and node.right is None:
          return node.label
        d = node.split_dim
        if x[d] <= node.split_point:
          node = node.left
        else:
          node = node.right
      return -1

    preds = [predict_one(x) for x in test_data]
    return preds

  """
  Students are encouraged to implement as many additional methods as they find helpful in completing
  the assignment. These methods can be implemented either as class methods of the Solution class or as
  global methods, depending on design preferences.

  For instance, one essential method that must be implemented is a method to build out the decision tree recursively.
  """


def _parse_row_tokens(row: str):
  """Parse a line of the form 'label idx1:val1 idx2:val2 ...' into (label:int, features: Dict[int,float])"""
  parts = row.strip().split()
  if not parts:
    return None, {}
  lab = int(parts[0])
  feats = {}
  for tok in parts[1:]:
    if ':' not in tok:
      continue
    idx_s, val_s = tok.split(':', 1)
    try:
      idx = int(idx_s)
      val = float(val_s)
      feats[idx] = val
    except Exception:
      continue
  return lab, feats


def _lines_to_matrix(lines):
  """Convert list of dataset lines into (labels, data_matrix) where data_matrix is List[List[float]].
  Feature indices in input start at 0. We make dense vectors up to max index present.
  """
  parsed = [(_parse_row_tokens(l)) for l in lines if l.strip()]
  labels = [p[0] for p in parsed]
  feats_list = [p[1] for p in parsed]
  max_idx = -1
  for f in feats_list:
    if f:
      local_max = max(f.keys())
      if local_max > max_idx:
        max_idx = local_max
  D = max_idx + 1 if max_idx >= 0 else 0
  data = []
  for f in feats_list:
    row = [0.0] * D
    for k, v in f.items():
      row[k] = v
    data.append(row)
  return labels, data


def _format_node(node: Node) -> str:
  return f"{'{'}split_dim: {node.split_dim}, split_point: {node.split_point:.4f}, label: {node.label}{'}'}"


def _preorder(node: Node):
  if node is None:
    return []
  out = [node]
  if node.left is not None:
    out += _preorder(node.left)
  if node.right is not None:
    out += _preorder(node.right)
  return out


def _inorder(node: Node):
  if node is None:
    return []
  out = []
  if node.left is not None:
    out += _inorder(node.left)
  out.append(node)
  if node.right is not None:
    out += _inorder(node.right)
  return out


def _is_split_info_input(lines):
  # split_info inputs: first line integer (split_dim), second line float (split_point), then data lines
  if len(lines) < 3:
    return False
  first = lines[0].strip()
  second = lines[1].strip()
  # data lines contain ':' for features; split_point line will not
  if ':' in second:
    return False
  try:
    _ = int(first)
    _ = float(second)
    return True
  except Exception:
    return False


def _run_from_stdin():
  data = sys.stdin.read().strip().splitlines()
  if not data:
    return

  sol = Solution()

  if _is_split_info_input(data):
    split_dim = int(data[0].strip())
    split_point = float(data[1].strip())
    labels, mat = _lines_to_matrix(data[2:])
    val = sol.split_info(mat, labels, split_dim, split_point)
    print(f"{val:.4f}")
    return

  # otherwise lines are dataset lines; detect classification (test rows with label -1)
  labels_all, mat_all = _lines_to_matrix(data)
  # find index of first -1; tests are those with label == -1
  train_lines = []
  test_lines = []
  for lab, row in zip(labels_all, mat_all):
    if lab == -1:
      test_lines.append(row)
    else:
      train_lines.append((lab, row))

  if test_lines:
    # classification task
    train_labels = [t for t, _ in train_lines]
    train_data = [r for _, r in train_lines]
    preds = sol.classify(train_data, train_labels, test_lines)
    for p in preds:
      print(int(p))
    return

  # otherwise tree structure task
  train_labels = labels_all
  train_data = mat_all
  sol.fit(train_data, train_labels)
  pre = _preorder(sol.root)
  ino = _inorder(sol.root)
  print(''.join(_format_node(n) for n in pre))
  print(''.join(_format_node(n) for n in ino))


if __name__ == "__main__":
  _run_from_stdin()