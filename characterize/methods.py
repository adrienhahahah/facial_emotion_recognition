from skimage import feature
import numpy as np
import cv2
import os
import numpy as np
from rois import face_rois

class Lbp:
    blocks_size = {
        'EYE LEFT': (2, 2),
        'EYE RIGHT': (2, 2),
        'EYEBROW LEFT': (4, 1),
        'EYEBROW RIGHT': (4, 1),
        'INTER EYEBROW': (1, 1),
        'MOUTH': (4, 3)
    }
    radius = 0
    numPoints = 0
    faceROIs = None
    histogramVector = np.array([])

    # @param imagePath : the absolute path of image path
    def __init__(self, imagePath, radius=1, numPoints=8):
        self.radius = radius
        self.numPoints = numPoints
        self.faceROIs = face_rois.Face_ROIs(imagePath, method='lbp')
        self.Lbp_Histogram()

    # @param ROI : one resized ROI image numpy array
    # @param keyBlock : determines which key's value, namely the ROI
    # -> HISTO : returns a concatenated histogram of current ROI, ROI may divided into small ROIs according to self.block_size
    def One_ROI_Lbp_Histogram(self, ROI, keyBlock):
        (W, H) = self.blocks_size[keyBlock]
        (height, width) = ROI.shape
        step_w = int(width / W)
        step_h = int(height / H)

        HISTO = np.array([])

        for i in range(0, height, step_h):                      # traverse numpy matrix by line
            for j in range(0, width, step_w):                   # traverse numpy matrix by column
                block = ROI[i:i + step_h, j:j + step_w]
                lbp = feature.local_binary_pattern(block, self.numPoints, self.radius, method="default")
                values = lbp.ravel()
                (histogram, _) = np.histogram(values, bins=256)
                HISTO = np.concatenate((HISTO, histogram))
        return HISTO


    # calculates the final eigenvector consisit of each ROI's histograms.
    def Lbp_Histogram(self):
        if len(self.faceROIs.resizedROIs) == 0:
            print("image resized ROI dictionary is empty, fail to generate histogram")
            return
        for keyBlock, ROI in self.faceROIs.resizedROIs.items():
            hist = self.One_ROI_Lbp_Histogram(ROI, keyBlock)
            self.histogramVector = np.concatenate((self.histogramVector, hist))




