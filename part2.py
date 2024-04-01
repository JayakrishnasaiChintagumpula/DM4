from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(X, max_k):
    sse_inertia = []
    sse_manual = []
    model_sse_inertia = {}
    model_sse_manual = {}
    
    for clusters_num in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=clusters_num, random_state=12)
        preds = kmeans.fit_predict(X)
        inertia = kmeans.inertia_
        sse_inertia.append(inertia)
        
        # Manual SSE calculation
        sse = {}
        for i in range(len(preds)):
            point = X[i]
            centroid = kmeans.cluster_centers_[preds[i]]
            distance = (point - centroid) ** 2
            sse_contrib = np.sum(distance)
            sse[preds[i]] = sse.get(preds[i], 0) + sse_contrib
        
        sse_manual_value = sum(sse.values())
        sse_manual.append(sse_manual_value)
        
        model_sse_inertia[clusters_num] = inertia
        model_sse_manual[clusters_num] = sse_manual_value

    return sse_inertia, sse_manual, model_sse_inertia, model_sse_manual




def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """

    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    X, y = make_blobs(center_box=(-20, 20), n_samples=20, centers=5, random_state=12)

    # Create an additional array, for example, a unique identifier for each sample
    sample_ids = np.arange(1, len(X) + 1)
    dct = answers["2A: blob"] = [X, y,sample_ids]

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """

    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """

    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    _, sse_values, _, _ = fit_kmeans(X, 8)

    # Plotting the manually calculated SSE as a function of k
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 9), sse_values, marker='o', linestyle='-', color='green', label='SSE (Manual Calculation)')
    plt.title('Elbow Method for Optimal k (Manual SSE Calculation)')
    plt.xlabel('Number of Clusters k')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.xticks(range(1, 9))
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(sse_values)
    dct = answers["2C: SSE plot"] = sse_values

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """

    # dct value has the same structure as in 2C
    _, _, model_sse_inertia, _ = fit_kmeans(X, 8)

    # Extracting the inertia values for plotting
    inertia_values = list(model_sse_inertia.values())
    
    # Plotting the inertia values as a function of k
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 9), inertia_values, marker='o', linestyle='-', color='blue', label='Inertia')
    plt.title('Inertia for Different Numbers of Clusters k')
    plt.xlabel('Number of Clusters k')
    plt.ylabel('Inertia (SSE)')
    plt.xticks(range(1, 9))
    plt.legend()
    plt.grid(True)
    plt.show()
    dct = answers["2D: inertia plot"] = inertia_values

    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
