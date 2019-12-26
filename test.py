# from rois.face_rois import Face_ROIs
import os
from characterize.methods import Lbp
import characterize.eigenvector as eigenvector
import characterize.svmModel as svmModel


path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path, 'lbpModels')
# t = Lbp(path).histogram

# path = os.path.dirname(os.path.abspath(__name__))


# vec_gen = eigenvector.Generator(databasePath='database', targetDirName='lbpVectors')
# vec_gen.readLbp2write()


# To execute the following, you now have already saved eigenvectors in the @param: eigenvectorsPath
# also you have to define the directory where models are going to be stored

# svmTrainer = svmModel.SvmTrainer(eigenvectorsPath='lbpVectors', targetModelPath='lbpModels')
# svmTrainer.read2Train()


classifier = svmModel.SVMClassifier(svmModelsPath='lbpModels')             #
# classifier.lbp_Predict(imagePath='images/an1.jpg')
# classifier.lbp_Predict(imagePath='images/ex.jpg')

precisionDict = classifier.test_Lbp_precision('images')
print(precisionDict)


print('hh')

