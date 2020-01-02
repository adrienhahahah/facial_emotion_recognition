import os
import numpy as np
from characterize import eigenvector
import characterize.svmModel as svmModel
from characterize import methods
import progressbar
import time


from IPython.display import clear_output

# dir = os.path.dirname(os.path.abspath(__file__))
# dir = os.path.join(dir, 'vectors')
#
vec_gen = eigenvector.Generator(databasePath='database', targetDirName='ltpVectors')
vec_gen.readLtp2write()

svmTrainer = svmModel.SvmTrainer(eigenvectorsPath='ltpVectors', targetModelPath='ltpModels')
svmTrainer.read2Train()
#
# start = time.time()
#
classifier = svmModel.SVMClassifier('ltpModels')
dict = classifier.test_Ltp_precision('jaffebase', display=True )
# print(dict)
# {'anger': 53.333, 'disgust': 48.276, 'fear': 3.125, 'happy': 48.387, 'sadness': 3.226, 'surprise': 86.667}
end = time.time()
# print(end - start)


classifier2 = svmModel.SVMClassifier('ltpModels')
dict2 = classifier2.test_Ltp_precision('database', display=True )
print(dict2)

end1 = time.time()
print('total time', end1 - end)

# aa = methods.Ltp('images/anger/an.jpg').histogramVector
# print(aa.shape, aa.dtype)

# p = progressbar.ProgressBar()
# p.start()
# for i in range(13):
#     p.update(i * 100 /13)
#     time.sleep(1)
# p.finish()




# for root, dirlist, files in os.walk(vect):
#     for i in range(len(files)):
#         for j in range(i + 1, len(files)):
#             a = files[i]
#             b = files[j]
#             a = a.split('.')[0]
#             b = b[:-4]
#             ff = a + '_' + b + 'SVM.dat'
#             print(ff)
#             file = os.path.join(root, files[i])
#             text = np.loadtxt(file, dtype='float32', delimiter=',')
#             print(text.dtype)

def meanX(dataX):
    return np.mean(dataX, axis=0)

#
# dir = os.path.dirname(os.path.abspath(__file__))
# dir = os.path.join(dir, 'lbpVectors')
#
# savePath = os.path.join(dir, 'anger.txt')
# vec = np.array([])
# vec = np.loadtxt(savePath, dtype='float32', delimiter=',')

#vec = tmp

#       (1365, 7424)

# for root, dirls, files in os.walk(dir):
#     for file in files:
#         filePath = os.path.join(root, file)
#         tmp = np.loadtxt(filePath, delimiter=',', dtype='float32')
#         if vec.size == 0:
#             vec = tmp
#         else:
#             vec = np.concatenate((vec, tmp), axis=0)
#
# print(vec.shape)
#
# with open(savePath, 'a') as f:
#     np.savetxt(f, vec, fmt='%.5f', delimiter=',')


#
# def pca(XMat, k):
#     average = meanX(XMat)
#     m, n = np.shape(XMat)
#     data_adjust = []
#     avgs = np.tile(average, (m, 1))
#     data_adjust = XMat - avgs
#     print(type(data_adjust))
#     covX = np.cov(data_adjust.T)   #计算协方差矩阵
#     featValue, featVec=  np.linalg.eig(covX)  #求解协方差矩阵的特征值和特征向量     # featVec.shape is (n, n), featValue.shape is (n,)
#     index = np.argsort(-featValue) #按照featValue进行从大到小排序                 # index.shape is (n,), specifying from big to small index of featvalue
#     finalData = []
#     if k > n:
#         print("k must lower than feature number")
#         return
#     else:
#         #注意特征向量时列向量，而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
#         selectVec = np.matrix(featVec.T[index[:k]]) #所以这里需要进行转置             # (k, n)
#         print(selectVec.shape)
#         #finalData = data_adjust * selectVec.T
#         #reconData = (finalData * selectVec) + average
#     return np.real(selectVec.T)
#


# # 单张图片测试，使用pca
# lbp = methods.Lbp('images/happy/ex.jpg')
# hist = lbp.histogramVector
# print(hist.dtype)
# selectVect = pca(vec, 2000)
# a = np.real(selectVect)
# maxreal = np.max(a)
# b = np.imag(selectVect)
# maximg = np.max(b)
# aver = meanX(vec)
# final_train = (hist - aver) * selectVect

print('hh')

# a,b = pca(vec, 5000)
# print(a.shape)          # 1365, 5000
# print(b.shape)          # 1365, 7424


# dt = np.array(ll)
# dt = meanX(dt)
# #dt = np.zeros((2,))
# print(dt)