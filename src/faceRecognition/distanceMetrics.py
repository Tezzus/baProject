""" distanceMetrics.py
has 2 different distance metrics for evaluating the similiarity between to face feature vectors.
cosine distance, L2,
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


def cosineDistance(features1, features2):
    """ calculates the cosine distance between two feature vectors. """
    return cosine_distances(features1.reshape(1, -1),features2.reshape(1, -1)).item()

def euclideanDistance(features1,features2):
    """calulates the euclidean distance between two feature vectors"""
    f1 = features1.reshape(1,-1)
    f2 = features2.reshape(1,-1)
    n1 = f1 / np.linalg.norm(f1)
    n2 = f2 / np.linalg.norm(f2)
    return euclidean_distances(n1,n2).item()
