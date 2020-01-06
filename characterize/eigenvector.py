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
                    imagePath = os.path.join(root, file)  # image absolute path
                    histogram = methods.Lbp(imagePath).histogramVector
                    print(file)
                    if len(histogram) == 0:
                        print("length of vector is 0, probably no face detected in this image:  ", file)
                        continue
                    with open(vectorFilePath, 'a') as f:
                        np.savetxt(f, [histogram], fmt='%.5f', delimiter=',')
                        print(file, " vector saved in ", vectorFilePath)

    # -> void: read image and generate LTP histogram as eigenvector, then save it in the self.targetPath under format .txt
    def readLtp2write(self):
        for root, dirlist, files in os.walk(self.database):
            emotion = os.path.basename(root)  # name of directory, which is emotion
            vectorFilePath = os.path.join(self.targetPath, emotion + '.txt')
            for file in files:
                    imagePath = os.path.join(root, file)  # image absolute path
                    histogram = methods.Ltp(imagePath).histogramVector
                    print(file)
                    if len(histogram) == 0:
                        print("length of vector is 0, probably no face detected in this image:  ", file)
                        continue
                    with open(vectorFilePath, 'a') as f:
                        np.savetxt(f, [histogram], fmt='%.5f', delimiter=',')
                        print(file, " vector saved in ", vectorFilePath)

    # -> void: read image and generate HOG histogram as eigenvector, then save it in the self.targetPath under format .txt
    def readHog2write(self):
        for root, dirlist, files in os.walk(self.database):
            emotion = os.path.basename(root)  # name of directory, which is emotion
            vectorFilePath = os.path.join(self.targetPath, emotion + '.txt')
            for file in files:
                imagePath = os.path.join(root, file)  # image absolute path
                histogram = methods.Hog(imagePath).histogramVector
                print(file)
                if len(histogram) == 0:
                    print("length of vector is 0, probably no face detected in this image:  ", file)
                    continue
                with open(vectorFilePath, 'a') as f:
                    np.savetxt(f, [histogram], fmt='%.5f', delimiter=',')
                    print(file, " vector saved in ", vectorFilePath)

    # -> void: read image and generate HOG+LTP histogram as eigenvector, then save it in the self.targetPath under format .txt
    def readHog_Ltp2write(self):
        for root, dirlist, files in os.walk(self.database):
            emotion = os.path.basename(root)  # name of directory, which is emotion
            vectorFilePath = os.path.join(self.targetPath, emotion + '.txt')
            for file in files:
                imagePath = os.path.join(root, file)  # image absolute path
                histogram = methods.Hog(imagePath).histogramVector
                histogram2 = methods.Ltp(imagePath).histogramVector
                histogram = np.append(histogram, histogram2)
                print(file)
                if len(histogram) == 0:
                    print("length of vector is 0, probably no face detected in this image:  ", file)
                    continue
                with open(vectorFilePath, 'a') as f:
                    np.savetxt(f, [histogram], fmt='%.5f', delimiter=',')
                    print(file, " vector saved in ", vectorFilePath)