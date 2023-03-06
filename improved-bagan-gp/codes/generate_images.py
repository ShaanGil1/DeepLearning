import os
import random
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Reshape, Dense, Dropout, \
    Activation, LeakyReLU, Conv2D, Conv2DTranspose, Embedding, \
    Concatenate, multiply, Flatten, BatchNormalization


n_classes = 4
latent_dim = 128
channel = 1

def plt_img(generator, epoch):
    np.random.seed(42)
    latent_gen = np.random.normal(size=(n_classes, latent_dim))

    x_real = x_test * 0.5 + 0.5
    n = n_classes

    plt.figure(figsize=(2*n, 2*(n+1)))
    for i in range(n):
        # display original
        ax = plt.subplot(n+1, n, i + 1)
        if channel == 3:
            plt.imshow(x_real[y_test==i][4].reshape(64, 64, channel))
        else:
            plt.imshow(x_real[y_test == i][4].reshape(64, 64))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for c in range(n):
            decoded_imgs = generator.predict([latent_gen, np.ones(n)*c])
            decoded_imgs = decoded_imgs * 0.5 + 0.5
            # display generation
            ax = plt.subplot(n+1, n, (i+1)*n + 1 + c)
            if channel == 3:
                plt.imshow(decoded_imgs[i].reshape(64, 64, channel))
            else:
                plt.imshow(decoded_imgs[i].reshape(64, 64))
                plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.savefig('bagan_gp_results/generated_plot_%d.png' % epoch)
    #plt.show()
    return


'''
classes are encoded in this format:

['glioma', 'meningioma', 'notumor', 'pituitary'] -> [0,1,2,3]
'''

#load generator
generator = tf.keras.models.load_model('brain_generator_5_.h5')
num_images = 10
target_class = 3
latent_gen = np.random.uniform(size=(n_classes, latent_dim))

decoded_imgs = generator.predict([latent_gen, np.ones(num_classes) * 5])
decoded_imgs = decoded_imgs * 0.5 + 0.5
for i in range(num_images):
  plt.imshow(decoded_imgs[i].reshape(64, 64))
  plt.show()

  