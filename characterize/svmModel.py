import cv2
import numpy as np
import os
from characterize import methods


class SvmTrainer:
    # the absolute directory path of eigenvector files
    eigenvectorsPath = None
    # the output svm model absolute directory path
    targetModelPath = None

    # @param eigenVectorsPath: the relative directory of current directory where has the eigenvector files
    # @param targetModelPath: the relative directory of current directory where the svm models will be stored
    def __init__(self, eigenvectorsPath, targetModelPath):
        currentPath = os.path.dirname(os.path.abspath(__name__))
        self.eigenvectorsPath = os.path.join(currentPath, eigenvectorsPath)
        self.targetModelPath = os.path.join(currentPath, targetModelPath)

    # -> void: read given self.eigenvectorPath path and save model .dat file to the self.targetModelPath
    def read2Train(self):
        for root, dirlist, files in os.walk(self.eigenvectorsPath):
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    emotionNamei = files[i][:-4]
                    emotionNamej = files[j][:-4]
                    svmFileName = emotionNamei + '_' + emotionNamej + 'SVM.dat'
                    svmFilePath = os.path.join(self.targetModelPath, svmFileName)
                    vectorFilePathi = os.path.join(root, files[i])  # image absolute path
                    vectorFilePathj = os.path.join(root, files[j])  # image absolute path
                    vectori = np.loadtxt(vectorFilePathi, dtype='float32', delimiter=',')
                    vectorj = np.loadtxt(vectorFilePathj, dtype='float32', delimiter=',')
                    trainVector = np.concatenate((vectori, vectorj), axis=0)
                    labels = np.zeros((vectori.shape[0] + vectorj.shape[0], 1), dtype=int)
                    labels[:vectori.shape[0]] = 1
                    labels[vectori.shape[0]:] = -1
                    # svm parameters
                    svm = cv2.ml.SVM_create()
                    svm.setKernel(cv2.ml.SVM_LINEAR)
                    svm.setType(cv2.ml.SVM_C_SVC)
                    svm.setC(0.3)

                    svm.train(trainVector, cv2.ml.ROW_SAMPLE, labels)
                    svm.save(svmFilePath)
                    print(svmFileName, ' saved at ', svmFilePath)



class SVMClassifier:
    __emotionsDict = {
        'anger': 0,
        'disgust': 0,
        'fear': 0,
        'happy': 0,
        'sadness': 0,
        'surprise': 0
    }
    svmModelsPath = None

    # @param svmModelsPath: provide the relative directory path where all the svm models are stored
    def __init__(self, svmModelsPath):
        currentDir = os.path.dirname(os.path.abspath(__name__))
        self.svmModelsPath = os.path.join(currentDir, svmModelsPath)

    # we need to reset the dictionary every time we execute the lbp_Predict()
    def reset_emotionsDict(self):
        for key in self.__emotionsDict.keys():
            self.__emotionsDict[key] = 0

    # @param imagePath: predict the emotion of the given imagePath
    # -> emotionPrediction: a str which indicates the emotion of the picture
    def lbp_Predict(self, imagePath):
        currentDir = os.path.dirname(os.path.abspath(__name__))
        imageAbsPath = os.path.join(currentDir, imagePath)
        imageLbpVector = methods.Lbp(imageAbsPath).histogramVector
        imageLbpVector = np.float32([imageLbpVector])                                 # convert image nparray to type float32
        for root, dirlist, files in os.walk(self.svmModelsPath):
            for file in files:
                emotions = file[:-7]
                emotions = emotions.split('_')                          # emotions = ['angry', 'happy']
                svmModelPath = os.path.join(root, file)
                svmModel = cv2.ml.SVM_load(svmModelPath)
                (_, predictVal) = svmModel.predict(imageLbpVector)
                if predictVal == 1:
                    self.__emotionsDict[emotions[0]] += 1
                else:
                    self.__emotionsDict[emotions[1]] += 1
        emotionPrediction = ''
        for key, value in self.__emotionsDict.items():
            if value == 5:
                emotionPrediction = key
        print(imagePath)
        print(self.__emotionsDict)
        print(emotionPrediction)
        self.reset_emotionsDict()
        return emotionPrediction

    def test_Lbp_precision(self, testImageBase):
        precisionDict = {}
        currentDir = os.path.dirname(os.path.abspath(__name__))
        testImageAbsBase = os.path.join(currentDir, testImageBase)
        for root, dirList, files in os.walk(testImageAbsBase):
            testNumber = len(files)
            correctCount = 0
            for file in files:
                testImageAbsPath = os.path.join(root, file)
                emotionPrediction = self.lbp_Predict(testImageAbsPath)
                emotionLabel = os.path.basename(root)
                if emotionPrediction == emotionLabel:
                    correctCount += 1
            if testNumber != 0:
                precision = round(correctCount / testNumber * 100, 3)
                print('Emotion ' + emotionLabel + ' precision is ', precision, '%')
                precisionDict[emotionLabel] = precision
        return precisionDict
