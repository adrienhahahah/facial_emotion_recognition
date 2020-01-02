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
    radius = 0          #
    numPoints = 0       # the number of points corresponding to the radius
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



class Ltp:
    blocks_size = {
        'EYE LEFT': (1, 2),
        'EYE RIGHT': (1, 2),
        'EYEBROW LEFT': (2, 1),
        'EYEBROW RIGHT': (2, 1),
        'INTER EYEBROW': (1, 1),
        'MOUTH': (3, 3)
    }

    faceROIs = None
    histogramVector = np.array([])

    def __init__(self, imagePath):
        self.faceROIs = face_rois.Face_ROIs(imagePath, method='ltp')
        self.Ltp_Histogram()


    # convert 8 bits of binary number in array to a decimal number
    def convertBinArray(self, vec):
        nb = 0
        for i in range(0, len(vec)):
            nb += vec[i] * (2 ** i)
        return nb

    # returns the 8 neighbours of the given coordinates
    # (it only gives the 8 values of gray)
    def neighbours(self, gray, coords):
        i, j = coords
        return np.array(
            [gray[i + 1, j], gray[i + 1, j + 1], gray[i, j + 1], gray[i - 1, j + 1], gray[i - 1, j], gray[i - 1, j - 1],
             gray[i, j - 1], gray[i + 1, j - 1]])

    # creates the vector s,
    # which is composed of 1,0,-1 depending of the difference between
    # the central pixel and its neighbours compared to the threshold t
    def computeS(self, gr_level_center, neighbrs, t):
        s = np.array([], dtype=int)
        for i in range(0, len(neighbrs)):
            val = 0
            x = gr_level_center - neighbrs[i]
            if x >= t:
                val = 1
            elif np.abs(x) < t:
                val = 0
            elif x <= -t:
                val = -1
            s = np.append(s, val)
        return s

    # creates two vectors from entry vector s,
    # the first is composed of the 0 and -1, and transfroms 1 into 0,
    # the second is composed of 0 and 1, and transform -1 into 1
    def computeNegativesPositives(self, s):
        negatives = np.array([], dtype=int)
        positives = np.array([], dtype=int)
        for i in range(0, len(s)):
            if s[i] == 1:
                negatives = np.append(negatives, 0)
                positives = np.append(positives, 1)
            elif s[i] == -1:
                negatives = np.append(negatives, 1)
                positives = np.append(positives, 0)
            else:
                negatives = np.append(negatives, 0)
                positives = np.append(positives, 0)
        return (negatives, positives)


    # returns the LTP vector of a given block:  'negative' LTP and 'positive' LTP
    def LTP(self, block):
        row, col = block.shape

        # add padding=1 of 0 to block
        block_pad = np.zeros((row + 2, col + 2))
        block_pad[1:row + 1, 1:col + 1] = block

        vecNegBase10 = np.array([], dtype=int)
        vecPosBase10 = np.array([], dtype=int)

        for i in range(1, row + 1):
            for j in range(1, col + 1):
                # gets all 8 neighbours around pixel at (i,j)
                neighrs = self.neighbours(block_pad, (i, j))

                # computes t value and create vector s composed of (-1,0,1) according to t
                t = np.mean(np.sqrt(neighrs))
                s = self.computeS(block_pad[i, j], neighrs, t)

                # creates both vectors witn (0,1) and (0,-1)
                negatives, positives = self.computeNegativesPositives(s)

                # converts both previous vectors from binaries to base 10, and puts the two new numbers into two vectors
                negBase10 = self.convertBinArray(negatives)
                posBase10 = self.convertBinArray(positives)
                vecNegBase10 = np.append(vecNegBase10, negBase10)
                vecPosBase10 = np.append(vecPosBase10, posBase10)

        # converts both vectors negatives and positives to histograms
        histoNeg = np.bincount(vecNegBase10, minlength=256)
        histoPos = np.bincount(vecPosBase10, minlength=256)

        return np.append(histoNeg, histoPos)

    # returns the LTP vector of one ROI. The ROI is divided in several blocks. From each block, it computes the 2
    def One_ROI_Ltp_Histogram(self, ROI, keyBlock):
        histos = np.array([], dtype=int)
        W, H = self.blocks_size[keyBlock]
        height, width = ROI.shape
        step_w = int(width / W)
        step_h = int(height / H)

        i = 0
        j = 0
        while i < height:
            j = 0
            while j < width:
                block = ROI[i:i + step_h, j:j + step_w]
                histos = np.append(histos, self.LTP(block))
                j += step_w
            i += step_h

        return histos

    def Ltp_Histogram(self):
        if len(self.faceROIs.resizedROIs) == 0:
            print("image resized ROI dictionary is empty, fail to generate histogram")
            return
        for keyBlock, ROI in self.faceROIs.resizedROIs.items():
            hist = self.One_ROI_Ltp_Histogram(ROI, keyBlock)
            self.histogramVector = np.concatenate((self.histogramVector, hist))


class Hog:
    blocks_size = {
        'EYE LEFT': (1,1),
        'EYE RIGHT': (1,1),
        'EYEBROW LEFT': (5,1),
        'EYEBROW RIGHT': (5,1),
        'INTER EYEBROW': (1, 1),
        'MOUTH': (5, 4)
    }

    faceROIs = None
    histogramVector = np.array([])

    def __init__(self, imagePath):
        self.