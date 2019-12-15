import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv


def get_training_data(training_file):

    # categories:
    # 0 --> Adidas
    # 1 --> Apple
    # 2 --> Coca Cola
    # 3 --> Nike

    with open(training_file) as training_file:
        content = training_file.readlines()
        content = [x.strip() for x in content] 

        training_labels= []
        prediction_coords = []
        training_imgs = []

        # image idx, path file, classifcation, x_min y_min x_max y_max
        for idx, line in enumerate(content):
            line = line.split()

            training_labels.append(line[2])

            img = cv.imread(line[1])
            dim = np.array(np.shape(img)[:2]) # gives dimensions of photo
            width = dim[1].astype(float)
            height = dim[0].astype(float)
            box_coords = np.array(line[3:])
            box_coords = box_coords.astype(float)
            box_coords = [box_coords[0] * width, box_coords[1] * height, box_coords[2] * width, box_coords[3] * height]
            prediction_coords.append(box_coords)
            training_imgs.append(img)

            # cv.rectangle(img, (int(box_coords[0]), int(box_coords[1])), (int(box_coords[2]), int(box_coords[3])), (255, 0, 0) , 2)

            # cv.imshow(line[0],img)
            # cv.waitKey(0)


    return training_imgs, training_labels, prediction_coords

def get_testing_data(test_file):
    with open(test_file) as test_file:
        content = test_file.readlines()
        content = [x.strip() for x in content] 

        test_imgs = []
        test_labels = []
        
        for line in content:
            line = line.split()
            img = cv.imread(line[1])
            test_imgs.append(img)
            test_labels.append(line[2])

    return test_imgs, test_labels

