"""main.py
"""
import numpy as np
import torch
from torch import nn
from src.faceRecognition.elasticFace import ElasticFace
from PIL import Image
from torchvision import transforms
import os
wPATH = os.getcwd()

"""all weights are obtained with elasticface arc+, place"""
weights = ['22744backbone.pth','79604backbone.pth','113720backbone.pth','170580backbone.pth','216068backbone.pth','272928backbone.pth','iresnet100-73e07ba7.pth']
elasticFace = ElasticFace()
elasticFace.setWeights(weights[5])

face1 = Image.open(os.path.join(wPATH,'samples/musk1.jpg'))
face2 = Image.open(os.path.join(wPATH,'samples/musk2.jpg'))

"""normalize faces - pixel values are between [-1,1]"""
mean = [0.5] * 3
std = [0.5 * 256 / 255] * 3
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

tensor1 = preprocess(face1)
tensor2 = preprocess(face2)

elasticFace.verifyFaces(tensor1,tensor2)


