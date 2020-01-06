# from rois.face_rois import Face_ROIs
import os
from characterize.methods import Lbp
import characterize.eigenvector as eigenvector
import characterize.svmModel as svmModel
import time
import pickle

#
# path = os.path.dirname(os.path.abspath(__file__))
# path1 = os.path.join(path, 'lbpPcaModels_PcaDict/PcaDict.pickle')
# path2 = os.path.join(path, 'lbpPcaModels_PcaSelectVec/PcaSelectVec.pickle')
#
# #
# dit1={}
# dit2={}


# t = Lbp(path).histogram

# path = os.path.dirname(os.path.abspath(__name__))


# vec_gen = eigenvector.Generator(databasePath='database', targetDirName='lbpVectors')
# vec_gen.readLbp2write()


# To execute the following, you now have already saved eigenvectors in the @param: eigenvectorsPath
# also you have to define the directory where models are going to be stored

# svmTrainer = svmModel.SvmTrainer(eigenvectorsPath='lbpVectors', targetModelPath='lbpPcaModels')
# svmTrainer.read2Train(k_pca=2000)
#
# start = time.time()
# classifier = svmModel.SVMClassifier(svmModelsPath='lbpModels')             #
# classifier.lbp_Predict(imagePath='images/anger/an1.jpg', meanPcaDictPath='lbpPcaModels_PcaDict/PcaDict.pickle', selectVecDictPath='lbpPcaModels_PcaSelectVec/PcaSelectVec.pickle')
# classifier.lbp_Predict(imagePath='images/ex.jpg')

#precisionDict = classifier.test_Lbp_precision(testImageBase='jaffebase', meanPcaDictPath='lbpPcaModels_PcaDict/PcaDict.pickle', selectVecDictPath='lbpPcaModels_PcaSelectVec/PcaSelectVec.pickle')
# {'anger': 93.333, 'disgust': 0.0, 'fear': 0.0, 'happy': 22.581, 'sadness': 29.032, 'surprise': 6.667}
# total time 3648.7426493167877
# precisionDict = classifier.test_Lbp_precision(testImageBase='jaffebase', display=True)
# #   {'anger': 73.333, 'disgust': 48.276, 'fear': 3.125, 'happy': 45.161, 'sadness': 22.581, 'surprise': 46.667}
# #   total time 1042.000097990036
# print(precisionDict)
# end = time.time()
# print("total time", end - start)



vec_gen = eigenvector.Generator(databasePath='base/train/', targetDirName='hog_ltpVectors')
vec_gen.readHog_Ltp2write()


svmTrainer = svmModel.SvmTrainer(eigenvectorsPath='hog_ltpVectors', targetModelPath='hog_ltpModel')
svmTrainer.read2Train()


print('hh')

