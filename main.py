from preprocess import get_training_data, get_testing_data
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, LeakyReLU, Dropout
from visualization import visualize


class Model(tf.keras.Model):
    def __init__(self):

        self.learning_rate = .001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.model = tf.keras.Sequential()

        # Block 1
        self.model.add(Conv2D(64, (7, 7), strides=(2, 2), padding='same', input_shape=[64, 64, 3]))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))

        # Block 2
        self.model.add(Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))

        # Block 3
        self.model.add(Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))

        # Block 4
        self.model.add(Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=251256, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))

        # Block 5
        self.model.add(Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=[3, 3], strides=[1, 1], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=[3, 3], strides=[2, 2], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))

        # Block 6
        self.model.add(Conv2D(filters=1024, kernel_size=[3, 3], strides=[3, 3], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Conv2D(filters=1024, kernel_size=[3, 3], strides=[3, 3], padding='same'))
        self.model.add(LeakyReLU(alpha=0.1))

        # Flatten Layer 1
        self.model.add(Flatten())

        # Dense Layer 1
        self.model.add(Dense(8192))
        self.model.add(LeakyReLU(alpha=0.1))

        self.model.add(Dropout(0.5))

        self.model.add(Dense(196, activation='linear'))


    def call(self, inputs):

        return self.model(inputs)

    def loss(self, logits, labels):

        return self.classification_loss(logits, labels) + \
            self.boxes_loss(logits, labels)
        

    def classification_loss(self, logits, labels):
        idx     = tf.range(0, 50, 1)
        p       = tf.gather(logits, indices=idx, axis=3)
        p_hat   = tf.gather(labels, indices=idx, axis=3)

        return tf.reduce_sum(tf.math.square(p - p_hat))
    
    def boxes_loss(self, logits, labels):
        xy      = tf.gather(logits, indices=[:4], axis=3) #hardcoded math stuff
        xy_hat  = tf.gather(labels, indices=[:4], axis=3)
        wh      = tf.gather(logits, indices=[5:], axis=3)
        wh_hat  = tf.gather(labels, indices=[5:], axis=3)
        

        return tf.reduce_sum(tf.math.square(xy - xy_hat)) + \
                  tf.reduce_sum(tf.math.square(wh - wh_hat))

    
def train(model, train_inputs, train_labels):

    losses = []
    
    with tf.GradientTape() as tape:
        predictions = model.call(train_inputs)
        loss = model.loss(predictions, train_labels)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    losses.append(loss)
    loss_avg = tf.reduce_mean(losses)

    return loss_avg


def test(model, testing_imgs, test_labels):
    predictions = model(testing_imgs)
    loss = model.loss(predictions, test_labels)
    visualize(model, testing_imgs, predictions, loss) # method is declared in visualize.py


def main():
    model = Model()

    training_labels, _ , training_imgs = get_training_data('train.txt')
    testing_imgs, test_labels = get_testing_data('test.txt') # test images

    model.train(model, training_imgs, training_labels)
    model.test(model, testing_imgs, test_labels) #includes visualization


if __name__ == '__main__':
    main()
