import pandas as pd
import numpy as np


def checker():
    original = pd.read_csv("data/monitor20.csv", header=None)
    reconstruction = pd.read_csv("data/output.csv", header=None)
    num_original = original.select_dtypes(include=[np.number])
    c_original = original.select_dtypes(exclude=[np.number])
    num_recon = reconstruction.select_dtypes(include=[np.number])
    c_recon = reconstruction.select_dtypes(exclude=[np.number])

    range = num_original.max() - num_original.min()
    percentages = (num_original-num_recon).abs().divide(range) * 100
    print("Max error threshold percentages per column: ")
    print(percentages.max().to_numpy())
    print("Categorical columns reconstructed successfully: ", c_original.equals(c_recon))


# def save_part():
#     monitor = pd.read_csv("data/monitor20.csv", header=None)
#     monitor = monitor.sample(frac=0.05)
#     monitor.to_csv('small.csv', index=False, header=False)


if __name__ == '__main__':
    checker()
