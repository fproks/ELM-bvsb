import numpy as np
import os
import cv2

class Pretreatment():
    @staticmethod
    def readGrayImageToDATA(rootPath: str) -> np.ndarray:
        assert os.path.exists(rootPath)
        assert os.path.isdir(rootPath)
        imageList=[]
        for i in os.listdir(rootPath):
            path=os.path.join(rootPath,i)

