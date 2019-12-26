import os
import numpy as np

dir = os.path.dirname(os.path.abspath(__file__))
vect = os.path.join(dir, 'vectors')

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


dt = np.zeros((2,1))
print(dt)