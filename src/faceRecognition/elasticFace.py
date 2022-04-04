"""elasticFace.py
Object that handels feature extraction and comparison it uses consine distance as default.
"""
from .backbones.iresnet import iresnet100
from .distanceMetrics import cosineDistance, euclideanDistance
import torch
import os

class ElasticFace:

    def __init__(self):
        self.backbone = iresnet100(num_features=512)
        self.distanceMetric = 'cosine' # default value

    def setWeights(self,wName):
        PATH = os.getcwd()
        self.backbone.load_state_dict(torch.load(os.path.join(PATH,'src/faceRecognition/backbones/weights/ElasticFace-Arc+/', wName), map_location=torch.device('cpu')))
        self.backbone.eval()

    def setDistanceMetric(self,metric):
        self.distanceMetric = metric

    def extractFeatures(self,imgTensor):
        return self.backbone(imgTensor.unsqueeze(0))[0]

    def verifyFaces(self,imgTensor1,imgTensor2):
        with torch.no_grad():
            features1 = self.extractFeatures(imgTensor1)
            features2 = self.extractFeatures(imgTensor2)
            ###Debuging
            print(f"Cosine Distance: %s" % cosineDistance(features1,features2))
            print(f"Euclidean Distance: %s" % euclideanDistance(features1, features2))
            if (self.distanceMetric == 'cosine'):
                return cosineDistance(features1,features2)
            elif (self.distanceMetric == 'L2'):
                return euclideanDistance(features1,features2)
            else:
                assert ('ERROR: DistanceMetric has to be set either to cosine or L2')
