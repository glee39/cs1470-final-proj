import cv2 as cv
import numpy as np
import tensorflow as tf
    

def visualize(model, imgs, predictions, loss):
    fake_var = 1
    class_names = ['Adidas','Apple','Coca Cola','Nike']

    for img in imgs:

        img = cv.imread(img)
        dim = np.array(np.shape(img)[:2]) # gives dimensions of photo
        width = dim[1].astype(float)
        height = dim[0].astype(float)

        logits = model(imgs)
        boxes, classes, scores = make_boxes(logits, 0.15, 0.15)


        for i in range(len(boxes)):

            class_name = class_names[classes[i]]
            score = scores[i]

            box_coords = boxes[i]
            x_min = int(box_coords[0] * width)
            y_min = int(box_coords[1] * height)
            x_max = int(box_coords[2] * width)
            y_max = int(box_coords[3] * height)

            img = cv.rectangle(img, str(class_name + " " + score), (x_min, y_min), (x_max, y_max), (255, 0, 0) , 2)

            cv.imshow("~DETECTION~",img)
            cv.waitKey(0)

def make_boxes(logits, confidence_threshold, score_threshold):

    boxes = []
    classes = []
    scores = []

    # Compute mask to picking bounding box in each cell
    conf1 = logits[:, :, 4][:, :, np.newaxis]
    conf2 = logits[:, :, 9][:, :, np.newaxis]
    conf = np.concatenate((conf1, conf2),axis=-1)
    mask = (conf == conf.max()) + (conf >= confidence_threshold)

    cell_size = 1./4

    for i in range(4):
        for j in range(4):
            for b in range(2):
                if mask[i, j, b]:
                    box = logits[i, j, b * 5: b * 5 + 4]
                    classes = logits[i, j, b * 5 + 4]

                    # Compute the offset of the cell
                    # Convert the center of bbx to image coordinates system
                    offset = np.array([j, i] * cell_size)
                    box[:2] *= cell_size + offset
                    box_coords = np.zeros_like(box)

                    # Compute the upper left and bottom right coordinates of bbx
                    box_coords[:2] = box[:2] - 0.5 * box[2:]
                    box_coords[2:] = box[:2] + 0.5 * box[2:]

                    # Get max prob and corresponding class index
                    max_prob = np.max(logits[i, j, 10:])
                    cls_index = np.argmax(logits[i, j, 10:])

                    if classes * max_prob > score_threshold:
                        boxes.append(box_coords)
                        classes.append(cls_index)
                        scores.append(classes * max_prob)

    return boxes, classes, scores
