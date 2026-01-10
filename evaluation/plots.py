import matplotlib.pyplot as plt
import numpy as np

def entropy_histogram(entropy_vals, save_path):
    plt.hist(entropy_vals, bins=30)
    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.savefig(save_path)
    plt.close()
