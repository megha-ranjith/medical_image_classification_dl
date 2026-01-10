import numpy as np

def entropy(prob):
    return -np.sum(prob * np.log2(prob + 1e-9), axis=1)

def apply_rejection(probs, threshold):
    ent = entropy(probs)
    accepted = ent <= threshold
    return accepted, ent
