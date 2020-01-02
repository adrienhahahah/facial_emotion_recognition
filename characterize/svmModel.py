import cv2
import numpy as np
import os
import pickle
import progressbar
from characterize import methods


class SvmTrainer:
    # the absolute directory path of eigenvector files
    eigenvectorsPath = None
    # the output svm model absolute directory path
    targetModelPath = None

    pca = None

    # @param eigenVectorsPath: the relative directory of current directory where has the eigenvector files
    # @param targetModelPath: the relative directory of current directory where the svm models will be stored
    def __init__(self, eigenvectorsPath, targetModelPath):
        currentPath = os.path.dirname(os.path.abspath(__name__))
        self.eigenvectorsPath = os.path.join(currentPath, eigenvectorsPath)
        self.targetModelPath = os.path.join(currentPath, targetModelPath)

    # -> void: read given self.eigenvectorPath path and save model .dat file to the self.targetModelPath
    def read2Train(self, k_pca=0):
        if k_pca != 0:
            self.pca = PCA(k_pca)
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
                    if k_pca != 0:
                        self.pca.set_NormalizationDict(svmFileName, trainVector)
                        selectVec = self.pca.pca_newR(trainVector, k_pca)
                        self.pca.set_SelectVecDict(svmFileName, selectVec)
                        trainVector = trainVector * selectVec
                        trainVector = trainVector.astype(np.float32)
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
                    print(svmFileName, ' saved at ', svmFilePath, flush=True)
        if k_pca != 0:
            pcaDictSavePath = os.path.join(os.path.dirname(self.targetModelPath), os.path.basename(self.targetModelPath)+'_PcaDict/PcaDict.pickle')
            with open(pcaDictSavePath, 'wb') as f:
                pickle.dump(self.pca.normalizationDict, f)
            pcaSelectVecSavePath = os.path.join(os.path.dirname(self.targetModelPath), os.path.basename(self.targetModelPath)+'_PcaSelectVec/PcaSelectVec.pickle')
            with open(pcaSelectVecSavePath, 'wb') as f:
                pickle.dump(self.pca.selectVectDict, f)




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
    def lbp_Predict(self, imagePath, meanPcaDictPath=None, selectVecDictPath=None, display=True):
        currentDir = os.path.dirname(os.path.abspath(__name__))
        imageAbsPath = os.path.join(currentDir, imagePath)
        imageLbpVector = methods.Lbp(imageAbsPath).histogramVector
        imageLbpVector = np.float32([imageLbpVector])                                 # convert image nparray to type float32
        meanPcaDict = {}
        selectVecDict = {}
        if meanPcaDictPath is not None and selectVecDictPath is not None:
            meanPcaDictPath = os.path.join(currentDir, meanPcaDictPath)
            selectVecDictPath = os.path.join(currentDir, selectVecDictPath)
            with open(meanPcaDictPath, 'rb') as f:
                meanPcaDict = pickle.load(f)
            with open(selectVecDictPath, 'rb') as f:
                selectVecDict = pickle.load(f)
        for root, dirlist, files in os.walk(self.svmModelsPath):
            for file in files:
                emotions = file[:-7]
                emotions = emotions.split('_')                          # emotions = ['anger', 'happy']
                svmModelPath = os.path.join(root, file)
                svmModel = cv2.ml.SVM_load(svmModelPath)
                adjustImageLbpVector = imageLbpVector
                if len(meanPcaDict) != 0 and len(selectVecDict) != 0:
                    meanPca = meanPcaDict[file]
                    selectVec = selectVecDict[file].astype('float32')
                    adjustImageLbpVector = imageLbpVector - meanPca
                    adjustImageLbpVector = adjustImageLbpVector * selectVec
                (_, predictVal) = svmModel.predict(adjustImageLbpVector)
                if predictVal == 1:
                    self.__emotionsDict[emotions[0]] += 1
                else:
                    self.__emotionsDict[emotions[1]] += 1
        emotionPrediction = ''
        for key, value in self.__emotionsDict.items():
            if value == 5:
                emotionPrediction = key
        if display:
            print(imagePath)
            print(self.__emotionsDict)
            print(emotionPrediction)
        self.reset_emotionsDict()
        return emotionPrediction

    def test_Lbp_precision(self, testImageBase, meanPcaDictPath=None, selectVecDictPath=None, display=False):
        precisionDict = {}
        currentDir = os.path.dirname(os.path.abspath(__name__))
        testImageAbsBase = os.path.join(currentDir, testImageBase)
        p = progressbar.ProgressBar()
        p.start()
        nbTestPhotos = 0
        fileCount = 0
        for root, dirList, files in os.walk(testImageAbsBase):
            nbTestPhotos += len(files)
        for root, dirList, files in os.walk(testImageAbsBase):
            testNumber = len(files)
            correctCount = 0
            for file in files:
                testImageAbsPath = os.path.join(root, file)
                emotionPrediction = self.lbp_Predict(testImageAbsPath, meanPcaDictPath, selectVecDictPath, display)
                emotionLabel = os.path.basename(root)
                fileCount += 1
                if emotionPrediction == emotionLabel:
                    correctCount += 1
                p.update(fileCount * 100 / nbTestPhotos)
            if testNumber != 0:
                precision = round(correctCount / testNumber * 100, 3)
                print('Emotion ' + emotionLabel + ' precision is ', precision, '%')
                precisionDict[emotionLabel] = precision
        p.finish()
        return precisionDict

    def ltp_Predict(self, imagePath, meanPcaDictPath=None, selectVecDictPath=None, display=True):
        currentDir = os.path.dirname(os.path.abspath(__name__))
        imageAbsPath = os.path.join(currentDir, imagePath)
        imageLtpVector = methods.Ltp(imageAbsPath).histogramVector
        imageLtpVector = np.float32([imageLtpVector])                                 # convert image nparray to type float32
        meanPcaDict = {}
        selectVecDict = {}
        if meanPcaDictPath is not None and selectVecDictPath is not None:
            meanPcaDictPath = os.path.join(currentDir, meanPcaDictPath)
            selectVecDictPath = os.path.join(currentDir, selectVecDictPath)
            with open(meanPcaDictPath, 'rb') as f:
                meanPcaDict = pickle.load(f)
            with open(selectVecDictPath, 'rb') as f:
                selectVecDict = pickle.load(f)
        for root, dirlist, files in os.walk(self.svmModelsPath):
            for file in files:
                emotions = file[:-7]
                emotions = emotions.split('_')                          # emotions = ['anger', 'happy']
                svmModelPath = os.path.join(root, file)
                svmModel = cv2.ml.SVM_load(svmModelPath)
                adjustImageLtpVector = imageLtpVector
                if len(meanPcaDict) != 0 and len(selectVecDict) != 0:
                    meanPca = meanPcaDict[file]
                    selectVec = selectVecDict[file].astype('float32')
                    adjustImageLtpVector = imageLtpVector - meanPca
                    adjustImageLtpVector = adjustImageLtpVector * selectVec
                (_, predictVal) = svmModel.predict(adjustImageLtpVector)
                if predictVal == 1:
                    self.__emotionsDict[emotions[0]] += 1
                else:
                    self.__emotionsDict[emotions[1]] += 1
        emotionPrediction = ''
        for key, value in self.__emotionsDict.items():
            if value == 5:
                emotionPrediction = key
        if display:
            print(imagePath)
            print(self.__emotionsDict)
            print(emotionPrediction)
        self.reset_emotionsDict()
        return emotionPrediction



    def test_Ltp_precision(self, testImageBase, meanPcaDictPath=None, selectVecDictPath=None, display=False):
        precisionDict = {}
        currentDir = os.path.dirname(os.path.abspath(__name__))
        testImageAbsBase = os.path.join(currentDir, testImageBase)
        p = progressbar.ProgressBar()
        p.start()
        nbTestPhotos = 0
        fileCount = 0
        for root, dirList, files in os.walk(testImageAbsBase):
            nbTestPhotos += len(files)
        for root, dirList, files in os.walk(testImageAbsBase):
            testNumber = len(files)
            correctCount = 0
            for file in files:
                testImageAbsPath = os.path.join(root, file)
                emotionPrediction = self.ltp_Predict(testImageAbsPath, meanPcaDictPath, selectVecDictPath, display)
                emotionLabel = os.path.basename(root)
                fileCount += 1
                if emotionPrediction == emotionLabel:
                    correctCount += 1
                p.update(fileCount * 100 / nbTestPhotos)
            if testNumber != 0:
                precision = round(correctCount / testNumber * 100, 3)
                print('Emotion ' + emotionLabel + ' precision is ', precision, '%')
                precisionDict[emotionLabel] = precision
        p.finish()
        return precisionDict


    def hog_Predict(self, imagePath, meanPcaDictPath=None, selectVecDictPath=None, display=True):
        currentDir = os.path.dirname(os.path.abspath(__name__))
        imageAbsPath = os.path.join(currentDir, imagePath)
        imageHogVector = methods.Hog(imageAbsPath).histogramVector
        imageHogVector = np.float32([imageHogVector])                                 # convert image nparray to type float32
        meanPcaDict = {}
        selectVecDict = {}
        if meanPcaDictPath is not None and selectVecDictPath is not None:
            meanPcaDictPath = os.path.join(currentDir, meanPcaDictPath)
            selectVecDictPath = os.path.join(currentDir, selectVecDictPath)
            with open(meanPcaDictPath, 'rb') as f:
                meanPcaDict = pickle.load(f)
            with open(selectVecDictPath, 'rb') as f:
                selectVecDict = pickle.load(f)
        for root, dirlist, files in os.walk(self.svmModelsPath):
            for file in files:
                emotions = file[:-7]
                emotions = emotions.split('_')                          # emotions = ['anger', 'happy']
                svmModelPath = os.path.join(root, file)
                svmModel = cv2.ml.SVM_load(svmModelPath)
                adjustImageHogVector = imageHogVector
                if len(meanPcaDict) != 0 and len(selectVecDict) != 0:
                    meanPca = meanPcaDict[file]
                    selectVec = selectVecDict[file].astype('float32')
                    adjustImageHogVector = imageHogVector - meanPca
                    adjustImageHogVector = adjustImageHogVector * selectVec
                (_, predictVal) = svmModel.predict(adjustImageHogVector)
                if predictVal == 1:
                    self.__emotionsDict[emotions[0]] += 1
                else:
                    self.__emotionsDict[emotions[1]] += 1
        emotionPrediction = ''
        for key, value in self.__emotionsDict.items():
            if value == 5:
                emotionPrediction = key
        if display:
            print(imagePath)
            print(self.__emotionsDict)
            print(emotionPrediction)
        self.reset_emotionsDict()
        return emotionPrediction


    def test_Hog_precision(self, testImageBase, meanPcaDictPath=None, selectVecDictPath=None, display=False):
        precisionDict = {}
        currentDir = os.path.dirname(os.path.abspath(__name__))
        testImageAbsBase = os.path.join(currentDir, testImageBase)
        p = progressbar.ProgressBar()
        p.start()
        nbTestPhotos = 0
        fileCount = 0
        for root, dirList, files in os.walk(testImageAbsBase):
            nbTestPhotos += len(files)
        for root, dirList, files in os.walk(testImageAbsBase):
            testNumber = len(files)
            correctCount = 0
            for file in files:
                testImageAbsPath = os.path.join(root, file)
                emotionPrediction = self.hog_Predict(testImageAbsPath, meanPcaDictPath, selectVecDictPath, display)
                emotionLabel = os.path.basename(root)
                fileCount += 1
                if emotionPrediction == emotionLabel:
                    correctCount += 1
                p.update(fileCount * 100 / nbTestPhotos)
            if testNumber != 0:
                precision = round(correctCount / testNumber * 100, 3)
                print('Emotion ' + emotionLabel + ' precision is ', precision, '%')
                precisionDict[emotionLabel] = precision
        p.finish()
        return precisionDict

    def hog_ltp_Predict(self, imagePath, meanPcaDictPath=None, selectVecDictPath=None, display=True):
        currentDir = os.path.dirname(os.path.abspath(__name__))
        imageAbsPath = os.path.join(currentDir, imagePath)
        imageLtpVector = methods.Ltp(imageAbsPath).histogramVector
        imageHogVector = methods.Hog(imageAbsPath).histogramVector
        imageVector = np.append(imageHogVector, imageLtpVector)
        imageVector = np.float32([imageVector])                                 # convert image nparray to type float32
        meanPcaDict = {}
        selectVecDict = {}
        if meanPcaDictPath is not None and selectVecDictPath is not None:
            meanPcaDictPath = os.path.join(currentDir, meanPcaDictPath)
            selectVecDictPath = os.path.join(currentDir, selectVecDictPath)
            with open(meanPcaDictPath, 'rb') as f:
                meanPcaDict = pickle.load(f)
            with open(selectVecDictPath, 'rb') as f:
                selectVecDict = pickle.load(f)
        for root, dirlist, files in os.walk(self.svmModelsPath):
            for file in files:
                emotions = file[:-7]
                emotions = emotions.split('_')                          # emotions = ['anger', 'happy']
                svmModelPath = os.path.join(root, file)
                svmModel = cv2.ml.SVM_load(svmModelPath)
                adjustImageVector = imageVector
                if len(meanPcaDict) != 0 and len(selectVecDict) != 0:
                    meanPca = meanPcaDict[file]
                    selectVec = selectVecDict[file].astype('float32')
                    adjustImageVector = imageVector - meanPca
                    adjustImageVector = adjustImageVector * selectVec
                (_, predictVal) = svmModel.predict(adjustImageVector)
                if predictVal == 1:
                    self.__emotionsDict[emotions[0]] += 1
                else:
                    self.__emotionsDict[emotions[1]] += 1
        emotionPrediction = ''
        for key, value in self.__emotionsDict.items():
            if value == 5:
                emotionPrediction = key
        if display:
            print(imagePath)
            print(self.__emotionsDict)
            print(emotionPrediction)
        self.reset_emotionsDict()
        return emotionPrediction


    def test_Hog_Ltp_precision(self, testImageBase, meanPcaDictPath=None, selectVecDictPath=None, display=False):
        precisionDict = {}
        currentDir = os.path.dirname(os.path.abspath(__name__))
        testImageAbsBase = os.path.join(currentDir, testImageBase)
        p = progressbar.ProgressBar()
        p.start()
        nbTestPhotos = 0
        fileCount = 0
        for root, dirList, files in os.walk(testImageAbsBase):
            nbTestPhotos += len(files)
        for root, dirList, files in os.walk(testImageAbsBase):
            testNumber = len(files)
            correctCount = 0
            for file in files:
                testImageAbsPath = os.path.join(root, file)
                emotionPrediction = self.hog_ltp_Predict(testImageAbsPath, meanPcaDictPath, selectVecDictPath, display)
                emotionLabel = os.path.basename(root)
                fileCount += 1
                if emotionPrediction == emotionLabel:
                    correctCount += 1
                p.update(fileCount * 100 / nbTestPhotos)
            if testNumber != 0:
                precision = round(correctCount / testNumber * 100, 3)
                print('Emotion ' + emotionLabel + ' precision is ', precision, '%')
                precisionDict[emotionLabel] = precision
        p.finish()
        return precisionDict


class PCA:
    kDimention = 0
    normalizationDict = {}
    selectVectDict = {}
    # dictSavePath = None

    def __init__(self, kDimention):
        # currentDir = os.path.dirname(os.path.abspath(__name__))
        # self.dictSavePath = os.path.join(currentDir, dictSavePath)
        self.kDimention = kDimention

    def set_NormalizationDict(self, svmFileName, trainVectors):
        self.normalizationDict[svmFileName] = self.mean_On_Dimention(trainVectors)

    def set_SelectVecDict(self, svmFileName, selectVec):
        self.selectVectDict[svmFileName] = selectVec

    set_NormalizationDict
    def mean_On_Dimention(self, trainVectors):
        return np.mean(trainVectors, axis=0)

    def pca_newR(self, XMat, k):
        average = self.mean_On_Dimention(XMat)
        m, n = np.shape(XMat)
        data_adjust = []
        avgs = np.tile(average, (m, 1))
        data_adjust = XMat - avgs
        covX = np.cov(data_adjust.T)                                                    # calculate co-variance matrix
        featValue, featVec = np.linalg.eig(covX)                                      # featVec.shape is (n, n), featValue.shape is (n,)
        index = np.argsort(-featValue)                                                  # index.shape is (n,), specifying from big to small index of featvalue
        finalData = []
        if k > n:
            print("k must lower than feature number")
            return
        else:
            selectVec = np.matrix(featVec.T[index[:k]])            # (k, n)
        return np.real(selectVec.T)