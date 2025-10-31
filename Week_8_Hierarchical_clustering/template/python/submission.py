# Submit this file to Gradescope
from typing import List
# you may use other Python standard libraries, but not data
# science libraries, such as numpy, scikit-learn, etc.

class Solution:
  def hclus_single_link(self, X: List[List[float]], K: int) -> List[int]:
    """Single link hierarchical clustering
    Args:
      - X: 2D input data
      - K: the number of output clusters
    Returns:
      A list of integers (range from 0 to K - 1) that represent class labels.
      The number does not matter as long as the clusters are correct.
      For example: [0, 0, 1] is treated the same as [1, 1, 0]"""
    return _agglomerative_clustering(X, K, linkage="single")

  def hclus_average_link(self, X: List[List[float]], K: int) -> List[int]:
    """Complete link hierarchical clustering"""
    return _agglomerative_clustering(X, K, linkage="average")

  def hclus_complete_link(self, X: List[List[float]], K: int) -> List[int]:
    """Average link hierarchical clustering"""
    return _agglomerative_clustering(X, K, linkage="complete")


def _euclidean(a, b):
  return sum((aa - bb) ** 2 for aa, bb in zip(a, b)) ** 0.5


def _agglomerative_clustering(X: List[List[float]], K: int, linkage: str = "single") -> List[int]:
  """Generic agglomerative clustering with three linkage types.

  Deterministic tie-breaking: when multiple pairs have the same distance,
  choose the pair whose smallest member index is smallest, then the other.
  """
  n = len(X)
  if K >= n:
    return list(range(n))
  if K <= 1:
    return [0] * n

  # Precompute pairwise point distances
  pdist = [[0.0] * n for _ in range(n)]
  for i in range(n):
    for j in range(i + 1, n):
      d = _euclidean(X[i], X[j])
      pdist[i][j] = d
      pdist[j][i] = d

  # clusters: list of lists of original indices
  clusters = [[i] for i in range(n)]

  def cluster_distance(ci, cj):
    # ci and cj are lists of indices
    if linkage == "single":
      # minimum pairwise distance
      best = float('inf')
      for a in ci:
        for b in cj:
          if pdist[a][b] < best:
            best = pdist[a][b]
      return best
    elif linkage == "complete":
      best = 0.0
      for a in ci:
        for b in cj:
          if pdist[a][b] > best:
            best = pdist[a][b]
      return best
    else:  # average
      s = 0.0
      cnt = 0
      for a in ci:
        for b in cj:
          s += pdist[a][b]
          cnt += 1
      return s / cnt if cnt else float('inf')

  # Agglomerate until we have K clusters
  while len(clusters) > K:
    best_d = float('inf')
    best_pair = None
    # find pair to merge
    for i in range(len(clusters)):
      for j in range(i + 1, len(clusters)):
        d = cluster_distance(clusters[i], clusters[j])
        if d < best_d - 1e-12:
          best_d = d
          best_pair = (i, j)
        elif abs(d - best_d) <= 1e-12:
          # tie-break deterministically by smallest indices
          cur = (min(clusters[i]), min(clusters[j]))
          prev = (min(clusters[best_pair[0]]), min(clusters[best_pair[1]]))
          if cur < prev:
            best_pair = (i, j)

    i, j = best_pair
    # merge j into i (keep order stable)
    new_cluster = clusters[i] + clusters[j]
    # remove j first (larger index) then replace i
    if j > i:
      del clusters[j]
      clusters[i] = new_cluster
    else:
      del clusters[i]
      clusters[j] = new_cluster

  # Build labels deterministically: sort clusters by smallest member index
  clusters_sorted = sorted(clusters, key=lambda c: min(c))
  labels = [None] * n
  for label, cluster in enumerate(clusters_sorted):
    for idx in cluster:
      labels[idx] = label
  return labels


def _run_from_stdin():
  import sys
  data = sys.stdin.read().strip().split()
  if not data:
    return
  it = iter(data)
  n = int(next(it))
  K = int(next(it))
  method = int(next(it))
  X = []
  for _ in range(n):
    x = float(next(it))
    y = float(next(it))
    X.append([x, y])

  sol = Solution()
  if method == 0:
    labels = sol.hclus_single_link(X, K)
  elif method == 1:
    labels = sol.hclus_average_link(X, K)
  else:
    labels = sol.hclus_complete_link(X, K)

  out = "\n".join(str(int(l)) for l in labels)
  sys.stdout.write(out)


if __name__ == "__main__":
  _run_from_stdin()
