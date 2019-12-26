from characterize import methods
import os
import numpy as np

class Generator:
    database = None
    targetPath = None


    # @param databasePath: the relative directory of current directory where has the marked database images.
    # @param method: a string indicating the method of generating eigenvector.
    # @param targetDirName: the relative directory of current directory where those eigenvector will be stored.
    def __init__(self, databasePath, targetDirName):
        currentPath = os.path.dirname(os.path.abspath(__name__))
        self.database = os.path.join(currentPath, databasePath)
        self.targetPath = os.path.join(currentPath, targetDirName)


    # -> void: read image and generate LBP histogram as eigenvector, then save it in the self.targetPath under format .txt
    def readLbp2write(self):
        for root, dirlist, files in os.walk(self.database):
            emotion = os.path.basename(root)  # name of directory, which is emotion
            vectorFilePath = os.path.join(self.targetPath, emotion + '.txt')
            for file in files:
                if file[6] == 'S' or file[0] == 's' or file[0] == 'S':
                    imagePath = os.path.join(root, file)  # image absolute path
                    histogram = methods.Lbp(imagePath).histogramVector
                    print(file)
                    if len(histogram) == 0:
                        print("length of vector is 0, probably no face detected in this image:  ", file)
                        continue
                    with open(vectorFilePath, 'a') as f:
                        np.savetxt(f, [histogram], fmt='%.5f', delimiter=',')
                        print(file, " vector saved in ", vectorFilePath)