""" distanceMetrics.py
has 2 different distance metrics for evaluating the similiarity between to face feature vectors.
cosine distance, L2,
"""
import numpy as np

def cosineDistance(features1, features2):
    """ calculates the cosine distance between two feature vectors. """
    feature1 = features1.cpu().numpy()
    feature2 = features2.cpu().numpy()
    n1 = np.linalg.norm(feature1)
    n2 = np.linalg.norm(feature2)
    x1 = feature1 / n1
    x2 = feature2 / n2
    return 1 - np.dot(x1,x2)

"""TODO Problem EUCLIDEAN DISTANCE"""

def euclideanDistance(features1,features2):
    """calulates the euclidean distance between two feature vectors"""
    ###with vector normalisation.
    feature1 = features1.detach().cpu()
    feature2 = features2.detach().cpu()
    n1 = np.linalg.norm(feature1)
    n2 = np.linalg.norm(feature2)

    x1 = feature1 / n1
    x2 = feature2 / n2
    return (x1 - x2).norm().item()
    """ With numpy
    feature1 = features1.cpu().numpy()
    feature2 = features2.cpu().numpy()
    diff = np.subtract(feature1, feature2)

    return np.linalg.norm(diff)
    """