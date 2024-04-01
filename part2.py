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

def fit_kmeans(X, n_clusters):
    """
    Fits a k-means clustering algorithm to the data X and explicitly calculates the SSE.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=12)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # Explicit calculation of SSE
    sse = 0
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        for point in cluster_points:
            sse += np.sum((point - centroids[i]) ** 2)
    return sse




def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """

    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    X, y = make_blobs(center_box=(-20, 20), n_samples=20, centers=5, random_state=12)
    dct = answers["2A: blob"] = [X, y]

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
    sse_values = []
    for k in range(1, 9):
        sse = fit_kmeans(X, k)
        sse_values.append(sse)

    # Plotting the SSE values to determine the elbow
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 9), sse_values, marker='o')
    plt.title('Elbow Method for Determining Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.xticks(range(1, 9))
    plt.grid(True)
    plt.show()

    return sse_values
    print(sse_values)
    dct = answers["2C: SSE plot"] = sse_values

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """

    # dct value has the same structure as in 2C
    inertia_values = []
    for k in range(1, 9):
        inertia = fit_kmeans_and_get_inertia(X, k)
        inertia_values.append(inertia)

    # Plotting the inertia values to determine the elbow
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 9), inertia_values, marker='o', color='red')
    plt.title('Elbow Method for Determining Optimal k using Inertia')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(range(1, 9))
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
