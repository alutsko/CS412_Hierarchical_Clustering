#!/usr/bin/env python3
"""Quick test of the Naive Bayes implementation"""

import sys
sys.path.insert(0, './template/python')
from submission import Solution

# Test case 1: Prior probabilities
# Expected: [0.2897, 0.1028, 0.1028, 0.1028, 0.0093, 0.1963, 0.1963]
X_train_1 = [
    [1,0,0,1,0,0,1,1,1,1,0,0,4,0,0,1],  # aardvark, class 1
    [0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0],  # worm, class 7
    [0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0],  # piranha, class 4
    [0,0,1,0,1,0,0,0,0,1,0,0,6,0,0,0],  # gnat, class 6
    [1,0,0,1,0,0,0,1,1,1,0,0,4,1,0,1],  # oryx, class 1
    [1,0,1,0,1,0,0,0,0,1,0,0,6,0,0,0],  # moth, class 6
    [0,1,1,0,1,1,1,0,1,1,0,0,2,1,0,0],  # skimmer, class 2
    [0,0,1,0,0,1,1,0,0,0,0,0,4,0,0,0],  # crab, class 7
    [1,0,0,1,1,0,0,1,1,1,0,0,2,1,0,0],  # vampire, class 1
    [0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0],  # slowworm, class 3
]

Y_train_1 = [1, 7, 4, 6, 1, 6, 2, 7, 1, 3]

# Test case 2: Classification
# Expected: [4] (Fish)
X_train_2 = X_train_1
Y_train_2 = Y_train_1
X_test_2 = [[0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0]]  # bass

sol = Solution()

# Test 1: Prior probabilities
print("Test 1: Prior probabilities")
priors = sol.prior(X_train_1, Y_train_1)
print(f"Computed: {[round(p, 4) for p in priors]}")
print(f"Expected: [0.2897, 0.1028, 0.1028, 0.1028, 0.0093, 0.1963, 0.1963]")
print()

# Test 2: Classification
print("Test 2: Classification")
labels = sol.label(X_train_2, Y_train_2, X_test_2)
print(f"Computed: {labels}")
print(f"Expected: [4]")
print()

# Verify Test 1
expected_priors = [0.2897, 0.1028, 0.1028, 0.1028, 0.0093, 0.1963, 0.1963]
match = all(abs(priors[i] - expected_priors[i]) < 0.0001 for i in range(7))
print(f"Test 1: {'PASS' if match else 'FAIL'}")

# Verify Test 2
print(f"Test 2: {'PASS' if labels == [4] else 'FAIL'}")
