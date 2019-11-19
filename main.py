import learner
from IntegralImage import IntegralImageRepresentation
import os
import random
import cv2
import sys
import time

def ReadImage(filepath, label):
    return IntegralImageRepresentation(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), label)

def GetImages(path, label):
    images = []
    for file_name in os.listdir(path):
        if file_name.endswith('png') or file_name.endswith('pgm'):
            images.append(ReadImage(os.path.join(path, file_name), label))
    return images

def LoadDatasetOfFaces(data_set_path, subdir):
    dir_path = os.path.join(data_set_path, subdir)
    faces = GetImages(os.path.join(dir_path, 'face'), 1)
    return faces
def LoadDatasetOfNonfaces(data_set_path, subdir):
    dir_path = os.path.join(data_set_path, subdir)
    non_faces = GetImages(os.path.join(dir_path, 'nonface'), -1)
    return non_faces

def main():
    if (len(sys.argv) != 2):
        print("You need to provide a dataset path")
        print("The dataset must contain subfolders 'test' and 'train'. Each subfolder must contain a directory with a folder of faces and a folder of nonfaces")
    else:
        data_set_path = sys.argv[1]
        faces = LoadDatasetOfFaces(data_set_path, 'train')
        non_faces = LoadDatasetOfNonfaces(data_set_path, 'train')

        T = 20

        # Each image is 24 x 24 pixels in our training/test dataset
        ada_boost_learner = learner.AdaBoost(image_height=24, image_width=24)
        ada_boost_learner.Learn(faces, non_faces, T)

        test_faces = LoadDatasetOfFaces(data_set_path, 'test')
        test_non_faces = LoadDatasetOfNonfaces(data_set_path, 'test')

        true_positives = 0
        true_negatives = 0
        test_data = test_faces + test_non_faces
        random.shuffle(test_data)

        for image in test_data:
            output = ada_boost_learner.TestOnImage(image)
            if image.label == 1 and output == 1:
                true_positives = true_positives + 1
            if image.label == -1 and output == -1:
                true_negatives = true_negatives + 1
        print("Correctly positive: " + str(true_positives) + " out of " + str(len(test_faces)))
        print("Correctly negative: " + str(true_negatives) + " out of " + str(len(test_non_faces)))

if __name__ == '__main__':
    main()