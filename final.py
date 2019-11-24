// This is the main file for our CS1470 Final Project, group Going Merry! 
// Premise: Implementation of a CNN model to perform logo detection
// goal accuracy: >= 70%

//CODE WILL BE REFACTORED LATER


from skimage.io import imread_collection


def preprocessing(file_path):

    dir = file_path
    imgs = imread_collection(col_dir)

    imgs = [tf.image.convert_image_dtype(i, dtype = tf.float32) for i in imgs] #convert each img to numbers
    imgs = [tf.convert_to_tensor(i) for i in imgs] #convert each img to a tensor

]

