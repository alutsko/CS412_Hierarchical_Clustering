# Submit this file to Gradescope
import math
from typing import Dict, List, Tuple
# You may use any built-in standard Python libraries
# You may NOT use any non-standard Python libraries such as numpy, scikit-learn, etc.

num_C = 7 # Represents the total number of classes

class Solution:
  
  def prior(self, X_train: List[List[int]], Y_train: List[int]) -> List[float]:
    """Calculate the prior probabilities of each class
    Args:
      X_train: Row i represents the i-th training datapoint
      Y_train: The i-th integer represents the class label for the i-th training datapoint
    Returns:
      A list of length num_C where num_C is the number of classes in the dataset
    """
    N = len(Y_train)
    priors = []
    
    # Calculate prior for each class (1 through num_C)
    for c in range(1, num_C + 1):
      count_c = Y_train.count(c)
      # P(y=c) = (count_c + 0.1) / (N + 0.1 * |C|)
      prior_c = (count_c + 0.1) / (N + 0.1 * num_C)
      priors.append(prior_c)
    
    return priors

  def label(self, X_train: List[List[int]], Y_train: List[int], X_test: List[List[int]]) -> List[int]:
    """Calculate the classification labels for each test datapoint
    Args:
      X_train: Row i represents the i-th training datapoint
      Y_train: The i-th integer represents the class label for the i-th training datapoint
      X_test: Row i represents the i-th testing datapoint
    Returns:
      A list of length M where M is the number of datapoints in the test set
    """
    # Get number of features
    num_features = len(X_train[0])
    
    # Define the possible values for each attribute
    # Most attributes are binary {0, 1}, but legs (index 12) has {0, 2, 4, 5, 6, 8}
    unique_values_per_attr = [2] * num_features
    unique_values_per_attr[12] = 6  # legs attribute has 6 possible values
    
    # Calculate priors
    priors = self.prior(X_train, Y_train)
    
    predictions = []
    
    for test_point in X_test:
      best_class = -1
      best_log_prob = float('-inf')
      
      # For each class, calculate P(y=c) * P(X|y=c)
      for c in range(1, num_C + 1):
        # Start with log prior
        log_prob = math.log(priors[c - 1])
        
        # Get all training points with class c
        class_indices = [i for i, y in enumerate(Y_train) if y == c]
        count_c = len(class_indices)
        
        # Multiply by conditional probabilities for each feature
        for feature_idx in range(num_features):
          feature_value = test_point[feature_idx]
          
          # Count how many training examples have class=c AND feature=feature_value
          count_feature_given_c = sum(
            1 for idx in class_indices if X_train[idx][feature_idx] == feature_value
          )
          
          # P(x_i=f|y=c) with Laplacian smoothing
          num_unique_values = unique_values_per_attr[feature_idx]
          prob_feature_given_c = (count_feature_given_c + 0.1) / (count_c + 0.1 * num_unique_values)
          
          # Add log probability
          log_prob += math.log(prob_feature_given_c)
        
        # Update best class if this is better
        if log_prob > best_log_prob:
          best_log_prob = log_prob
          best_class = c
      
      predictions.append(best_class)
    
    return predictions
