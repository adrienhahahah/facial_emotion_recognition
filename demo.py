import os
import cv2
import numpy as np
from characterize import eigenvector
import characterize.svmModel as svmModel
import imutils
from characterize import methods
import progressbar
import time
import argparse

# USAGE EXAMPLE
# python demo.py --image_path images/anger/an.jpg --models_path lbpModels

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--image_path", required=True,
	help="path to the image for test")
ap.add_argument("-m", "--models_path", required=True,
	help="path to the trained models")
args = vars(ap.parse_args())


classifier = svmModel.SVMClassifier(svmModelsPath=args['models_path'])             #
# classifier.lbp_Predict(imagePath='images/anger/an1.jpg', meanPcaDictPath='lbpPcaModels_PcaDict/PcaDict.pickle', selectVecDictPath='lbpPcaModels_PcaSelectVec/PcaSelectVec.pickle')
emotionDict, emotion = classifier.lbp_Predict(imagePath=args['image_path'])
image = cv2.imread(args["image_path"])
image = imutils.resize(image, width=500)
if emotion == '':
    outputStr = ''
    for key, val in emotionDict.items():
        if val == 4:
            outputStr = outputStr + key + ' '
    cv2.putText(image, outputStr, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
else:
    cv2.putText(image, emotion, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

# show the output image with the predicted emotion
cv2.imshow("Output", image)
cv2.waitKey(0)