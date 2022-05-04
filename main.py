"""main.py
"""
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.faceRecognition.elasticFace import ElasticFace
from PIL import Image
from src.FaceImage import FaceImage
from torchvision import transforms
from facenet_pytorch import MTCNN, extract_face
import os
wPATH = os.getcwd()
FACEREFPATH = "Test/Alexandra_Braun/FS_PE242148.jpg" #path to face reference image
GALLERYPATH = "Test/Alexandra_Braun" #path to gallery
help(MTCNN)
"""all weights are obtained with elasticface arc+, place"""
method = ["ElasticFace-Arc","ElasticFace-Arc+"]
# 0-5 ArcFace+ | 7 ArcFace
weights = ['22744backbone.pth','79604backbone.pth','113720backbone.pth','170580backbone.pth','216068backbone.pth','272928backbone.pth','iresnet100-73e07ba7.pth','295672backbone.pth']
listImages = []
elasticFace = ElasticFace()
elasticFace.setWeights(os.path.join(method[1],weights[5]))
mtcnn = MTCNN(image_size=112, margin=40,thresholds=[0.6, 0.7, 0.7], keep_all=True, post_process=False, device='cpu')

refImage = Image.open(os.path.join(wPATH,FACEREFPATH))

refFace = mtcnn(refImage)
#i = refFace / 250
#plt.imshow(i[0].permute(1, 2, 0))
#plt.show()
refFace = (refFace/255 - 0.5) / 0.5
refFace = FaceImage(True, refFace, refImage, FACEREFPATH,'refPhoto', None, elasticFace.extractFeatures(refFace))
faceGallery = []
faceBOXES = []
imagePATHlist = os.listdir(GALLERYPATH)
for imagePATH in imagePATHlist:
    if ".jpg" not in imagePATH and ".jpeg" not in imagePATH and ".png" not in imagePATH:
        continue #if it is not an image file -> ignore and continue
    img = Image.open(os.path.join(wPATH,GALLERYPATH,imagePATH))
    #DO PREPROCESSING
    faceBOX = mtcnn.detect(img,landmarks=False)
    for count, box in enumerate(faceBOX):
        if count >= len(faceBOX) / 2: break #just use faces not probabilities
        face = extract_face(img, box[0], image_size=112, margin=40)

        #i = img / 250
        #plt.imshow(i.permute(1, 2, 0))
        #plt.show()
        face = (face/255 - 0.5) / 0.5
        sImage = FaceImage(False, face, img, imagePATHlist, imagePATH, box[0], elasticFace.extractFeatures(face.unsqueeze(0)))
        listImages.append(sImage)
distances = elasticFace.verifyFaces(refFace,listImages)
print(distances)


