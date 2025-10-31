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
    return self._hclus_generic(X, K, measure="single")

  def hclus_average_link(self, X: List[List[float]], K: int) -> List[int]:
    """Complete link hierarchical clustering"""
    return self._hclus_generic(X, K, measure="average")

  def hclus_complete_link(self, X: List[List[float]], K: int) -> List[int]:
    """Average link hierarchical clustering"""
    return self._hclus_generic(X, K, measure="complete")

  def _hclus_generic(self, X: List[List[float]], K: int, measure: str) -> List[int]:
    """Generic agglomerative hierarchical clustering implementation.

    measure: one of 'single', 'complete', 'average'
    """
    import math

    n = len(X)
    if K <= 0 or K > n:
      raise ValueError("K must be between 1 and N")

    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
      xi = X[i]
      for j in range(i + 1, n):
        xj = X[j]
        d = math.hypot(xi[0] - xj[0], xi[1] - xj[1])
        dist[i][j] = d
        dist[j][i] = d

    clusters = [[i] for i in range(n)]

    def cluster_distance(a: List[int], b: List[int]) -> float:
      if measure == "single":
        best = float("inf")
        for i in a:
          for j in b:
            if dist[i][j] < best:
              best = dist[i][j]
        return best
      elif measure == "complete":
        best = 0.0
        for i in a:
          for j in b:
            if dist[i][j] > best:
              best = dist[i][j]
        return best
      elif measure == "average":
        s = 0.0
        cnt = 0
        for i in a:
          for j in b:
            s += dist[i][j]
            cnt += 1
        return s / cnt if cnt > 0 else float("inf")
      else:
        raise ValueError("unknown measure")

    while len(clusters) > K:
      best_pair = None
      best_d = float("inf")
      m = len(clusters)
      for i in range(m):
        for j in range(i + 1, m):
          d = cluster_distance(clusters[i], clusters[j])
          if d < best_d - 1e-12:
            best_d = d
            best_pair = (i, j)
      i, j = best_pair
      new_cluster = clusters[i] + clusters[j]
      if i < j:
        del clusters[j]
        del clusters[i]
      else:
        del clusters[i]
        del clusters[j]
      clusters.append(new_cluster)

    labels = [0] * n
    for cid, cluster in enumerate(clusters):
      for idx in cluster:
        labels[idx] = cid
    return labels
