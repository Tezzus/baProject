class FaceImage:

    def __init__(self, refFace, img, originalImage, originalPath, originalName,box, features):
        self.refFace = refFace
        self.img = img
        self.originalImage = originalImage
        self.originalPath = originalPath
        self.originalName = originalName
        self.box = box
        self.features = features
