import cv2
import dlib
import math
from imutils import face_utils
import os

class Face_ROIs:
    # properties
    image = 0
    grayImage = 0
    resizedROIs = {}
    method = 'lbp'
    ROI_scale = {   'lbp': {
                            'EYE LEFT' : (36, 24),
                            'EYE RIGHT': (36, 24),
                            'EYEBROW LEFT' : (80, 20),
                            'EYEBROW RIGHT': (80, 20),
                            'INTER EYEBROW' : (30, 30),
                            'MOUTH' : (80, 60)
                            },
    }
    __PREDICTOR_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../shape predictor/shape_predictor_68_face_landmarks.dat')

    def __init__(self, imagePath, method='lbp'):
        self.image = cv2.imread(imagePath)
        self.grayImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        FaceDetector = dlib.get_frontal_face_detector()                     # initialize facial detector
        faces = FaceDetector(self.grayImage, 1)                                  # return faces detected into list[] faces
        if len(faces) != 1:
            print("No face or more than one face in the image")
            self.resizedROIs = {}
        else:
            ShapePredictor = dlib.shape_predictor(self.__PREDICTOR_FILE_PATH)      # using predictor file to return 68 trait points on face
            shape = ShapePredictor(self.grayImage, faces[0])
            shape = face_utils.shape_to_np(shape)                            # convert shape object to numpy format
            originalROIs = self.get_Original_ROIs(shape)
            self.resize_ROIs(originalROIs)

    # @param shape: a numpy array (68,2), coordinates of traits on face
    # -> originalROIs{}: a dictionary of original coordinates of each facial traits     { 'EYEBROW LEFT' : xmin, ymin, xmax, ymax }
    def get_Original_ROIs(self, shape):
        originalROIs = {}
        #   LEFT EYEBROW    #
        xmin = math.inf
        ymin = math.inf
        xmax = 0
        ymax = 0
        for (x, y) in shape[22:27]:
            xmin = x if x < xmin else xmin
            xmax = x if x > xmax else xmax
            ymin = y if y < ymin else ymin
            ymax = y if y > ymax else ymax
        originalROIs["EYEBROW LEFT"] = (xmin, ymin, xmax, ymax)

        #   RIGHT EYEBROW    #
        xmin = math.inf
        ymin = math.inf
        xmax = 0
        ymax = 0
        for (x, y) in shape[17:22]:
            xmin = x if x < xmin else xmin
            xmax = x if x > xmax else xmax
            ymin = y if y < ymin else ymin
            ymax = y if y > ymax else ymax
        originalROIs["EYEBROW RIGHT"] = (xmin, ymin, xmax, ymax)

        #   INTER EYEBROW   #
        xmin = shape[21, 0]
        ymin = min(shape[17:27, 1])
        xmax = shape[22, 0]
        ymax = shape[27, 1]
        originalROIs["INTER EYEBROW"] = (xmin, ymin, xmax, ymax)

        #   EYE LEFT    #
        xmin = math.inf
        ymin = math.inf
        xmax = 0
        ymax = 0
        for (x, y) in shape[42:48]:
            xmin = x if x < xmin else xmin
            xmax = x if x > xmax else xmax
            ymin = y if y < ymin else ymin
            ymax = y if y > ymax else ymax
        originalROIs["EYE LEFT"] = (xmin, ymin, xmax, ymax)

        #   EYE RIGHT    #
        xmin = math.inf
        ymin = math.inf
        xmax = 0
        ymax = 0
        for (x, y) in shape[36:42]:
            xmin = x if x < xmin else xmin
            xmax = x if x > xmax else xmax
            ymin = y if y < ymin else ymin
            ymax = y if y > ymax else ymax
        originalROIs["EYE RIGHT"] = (xmin, ymin, xmax, ymax)

        #   MOUTH    #
        ymin = math.inf
        ymax = 0
        for (x, y) in shape[48:68]:
            ymin = y if y < ymin else ymin
            ymax = y if y > ymax else ymax
        originalROIs["MOUTH"] = (shape[41][0], ymin, shape[46][0], ymax)

        return originalROIs

    # @param originalROIs: a dictionary of original coordinates of each facial traits     { 'EYEBROW LEFT' : xmin, ymin, xmax, ymax }
    # @param traitKey： a key value which is one of the facial traits which corresponds to current self.method
    def resize_ROI(self, originalROIs, traitKey):
        (x1, y1, x2, y2) = originalROIs[traitKey]
        ROI_image = self.grayImage[y1:y2, x1:x2]
        return cv2.resize(ROI_image, self.ROI_scale[self.method][traitKey], interpolation=cv2.INTER_AREA)

    # @param originalROIs: a dictionary of original coordinates of each facial traits     { 'EYEBROW LEFT' : xmin, ymin, xmax, ymax }
    # -> self.resizedROIs: a dictionary of key which is facail traits, and values are image numpy array
    def resize_ROIs(self, originalROIs):
        for trait in self.ROI_scale[self.method].keys():
            self.resizedROIs[trait] = self.resize_ROI(originalROIs, trait)



