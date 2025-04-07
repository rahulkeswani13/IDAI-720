from collections import Counter
import numpy as np

def Reweighing(X, Y, A):
    # X: independent variables (2-d pd.DataFrame)
    # Y: the dependent variable (1-d np.array, binary y in {0,1})
    # A: a list/array of the names of the sensitive attributes with binary values
    # Return: sample_weight, an array of float weight for every data point
    #         sample_weight(a,y) = P(y)*P(a)/P(a,y)
    # Write your code below:
    # Get number of samples
    n = len(Y)
    
    # Create a combined attribute by concatenating all sensitive attributes ("male and white" or "female and asian")
    # This handles intersectionality of multiple sensitive attributes
    combined_attrs = []
    for i in range(n):
        attr_values = tuple(X[attr][i] for attr in A)
        combined_attrs.append(attr_values)
    
    # Calculate P(y)
    y_counts = Counter(Y)
    p_y = {y: count / n for y, count in y_counts.items()}
    
    # Calculate P(a) for combined attributes
    a_counts = Counter(combined_attrs)
    p_a = {a: count / n for a, count in a_counts.items()}
    
    # Calculate P(a,y) for each combination
    a_y_pairs = list(zip(combined_attrs, Y))
    a_y_counts = Counter(a_y_pairs)
    p_a_y = {pair: count / n for pair, count in a_y_counts.items()}
    
    # Calculate weights for each instance
    sample_weight = np.zeros(n)
    for i in range(n):
        a_val = combined_attrs[i]
        y_val = Y[i]
        
        # Apply the formula: P(y)*P(a)/P(a,y)
        weight = p_y[y_val] * p_a[a_val] / p_a_y[(a_val, y_val)]
        sample_weight[i] = weight

    # Rescale the sum of sample weights to len(y) before returning it
    sample_weight = sample_weight * len(Y) / sum(sample_weight)
    return sample_weight


