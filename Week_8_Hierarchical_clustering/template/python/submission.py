# Submit this file to Gradescope
from typing import List
import sys
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
    return _agg_clust(X, K, linkage="single")

  def hclus_average_link(self, X: List[List[float]], K: int) -> List[int]:
    """Complete link hierarchical clustering"""
    return _agg_clust(X, K, linkage="average")

  def hclus_complete_link(self, X: List[List[float]], K: int) -> List[int]:
    """Average link hierarchical clustering"""
    return _agg_clust(X, K, linkage="complete")


def EuclidDist(a, b):
  return sum((aa - bb) ** 2 for aa, bb in zip(a, b)) ** 0.5


def _agg_clust(X: List[List[float]], K: int, linkage: str = "single") -> List[int]:
  """Generic agglomerative clustering with three linkage types.

  Deterministic tie-breaking: when multiple pairs have the same distance,
  choose the pair whose smallest member index is smallest, then the other.
  """
  n = len(X)
  print("start", n, K, linkage, file=sys.stderr)
  # minimal trace: no per-point listing to reduce noise
  if K >= n:
    return list(range(n))
  if K <= 1:
    return [0] * n

  # Precompute pairwise point distances
  Pdist = [[0.0] * n for _ in range(n)]
  for i in range(n):
    for j in range(i + 1, n):
      d = EuclidDist(X[i], X[j])
      Pdist[i][j] = d
      Pdist[j][i] = d
  print("dists", file=sys.stderr)
  for i in range(n):
    for j in range(i+1, n):
      print(i, j, Pdist[i][j], file=sys.stderr)

  # clusters: list of lists of original indices
  clusters = [[i] for i in range(n)]

  def clusterDistance(ci, cj):
    # ci and cj are lists of indices
    if linkage == "single":
      # minimum pairwise distance
      best = float('inf')
      for a in ci:
        for b in cj:
          if Pdist[a][b] < best:
            best = Pdist[a][b]
      return best
    elif linkage == "complete":
      best = 0.0
      for a in ci:
        for b in cj:
          if Pdist[a][b] > best:
            best = Pdist[a][b]
      return best
    else:  # average
      s = 0.0
      cnt = 0
      for a in ci:
        for b in cj:
          s += Pdist[a][b]
          cnt += 1
      return s / cnt if cnt else float('inf')

  # Agglomerate until we have K clusters
  step = 0
  while len(clusters) > K:
    bestD = float('inf')
    bestPair = None
    # find pair to merge
    for i in range(len(clusters)):
      for j in range(i + 1, len(clusters)):
        d = clusterDistance(clusters[i], clusters[j])
        if d < bestD - 1e-12:
          bestD = d
          bestPair = (i, j)
        elif abs(d - bestD) <= 1e-12:
          # tie-break deterministically by smallest indices
          cur = (min(clusters[i]), min(clusters[j]))
          prev = (min(clusters[bestPair[0]]), min(clusters[bestPair[1]]))
          if cur < prev:
            bestPair = (i, j)

    i, j = bestPair
    # merge j into i (keep order stable)
    new_cluster = clusters[i] + clusters[j]
    print("step", step, "merge", i, j, bestD, file=sys.stderr)
    # remove j first (larger index) then replace i
    if j > i:
      del clusters[j]
      clusters[i] = new_cluster
    else:
      del clusters[i]
      clusters[j] = new_cluster
    step += 1

  # Build labels deterministically: sort clusters by smallest member index
  clustersSorted = sorted(clusters, key=lambda c: min(c))
  labels = [None] * n
  for label, cluster in enumerate(clustersSorted):
    for idx in cluster:
      labels[idx] = label
  return labels


def runFromStdin():
  import sys
  data = sys.stdin.read().strip().split()
  if not data:
    return
  it = iter(data)
  n = int(next(it))
  K = int(next(it))
  methodType = int(next(it))
  X = []
  for _ in range(n):
    x = float(next(it))
    y = float(next(it))
    X.append([x, y])

  sol = Solution()
  if methodType == 0:
    labels = sol.hclus_single_link(X, K)
  elif methodType == 1:
    labels = sol.hclus_average_link(X, K)
  else:
    labels = sol.hclus_complete_link(X, K)

  out = "\n".join(str(int(l)) for l in labels)
  sys.stdout.write(out)


if __name__ == "__main__":
  runFromStdin()
