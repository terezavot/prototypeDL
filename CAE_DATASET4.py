# python3
# -*- coding: utf-8 -*-
"""

@original_author: Oscar Li
@author of changes: Tereza Votypkova
"""

## TODO: find a correct level of abstraction, add some data preprocessing to the main code (wrapper function)
## TODO: make it possible to plug the wrappers into the main code (not to spend too much time on the preprocessing)
## TODO: matrices --> color code the smallest values (so we can see which ones are activated the most), make
## it more document friendly (smaller, more compact)
## TODO: why do we have the stripey thingies for the colors --> start training on more epochs with higher accuracy to see
## TODO: check if GPU is available on server, screens can be used --> it opens up a screen and it disconnects from the server
## TODO: search for template for thesis, start thinking about a structure
## 22nd --> 13:00


# noinspection PyUnresolvedReferences
from __future__ import division, print_function, absolute_import
import os
import time
import tensorflow as tf
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import (Dataset, DataLoader)
import random
import numpy as np
import matplotlib
import torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna

from autoencoder_helpers import makedirs, list_of_distances, print_and_write, list_of_norms
from data_preprocessing import batch_elastic_transform, batch_elastic_transform_color

import pandas as pd
from IPython.core.display import HTML
import glob
from typing import Tuple, List, Dict
import csv
import cv2
import PIL
from typing import Optional, Any
from collections import defaultdict
#### Import  data ####
######################
from tensorflow import keras
# writing a custom ImageFolder class to take care of one-hot encoding

class PrototypeDL():
    def __init__(self, data_path, color_channels, height, model_folder, flag):
        self.data_path = data_path
        self.flag = flag
        self.model_folder = model_folder
        self.color_channels = color_channels
        self.PATH = PreprocessData().preprocess_data(self.data_path, self.flag)
        self.batch_size = 32
        self.img_height = height
        self.img_width = self.img_height
        self.IMG_SIZE = (self.img_height, self.img_width)
        self.model_folder = os.path.join(os.getcwd(), "saved_model", "test_model", self.model_folder)
        self.prototype_folder = os.path.join(os.getcwd(), "saved_model", "test_model", self.model_folder, "prototypes")
        self.model_filename = "mnist_cae"
        self.n_saves = None
        # training parameters
        self.learning_rate = 0.002
        self.training_epochs = 70  # 1500
        # parameter to tune --> batch_size
        # batch_size = 250  # the size of a minibatch
        self.test_display_step = 100  # how many epochs we do evaluate on the test set once
        self.save_step = 3  # how frequently do we save the model to disk
        # elastic deformation parameters
        self.sigma = 4
        self.alpha = 20
        # lambda's are the ratios between the four error terms
        self.lambda_class = 20
        # parameters to tune
        self.lambda_ae = 1
        self.lambda_1 = 1  # 1 and 2 here corresponds to the notation we used in the paper
        self.lambda_2 = 1

        self.input_height = 28
        # input_height = 28  # MNIST data input shape
        self.input_width = self.input_height
        self.n_input_channel = color_channels  # the number of color channels; for MNIST is 1.
        self.input_size = self.input_height * self.input_width * self.n_input_channel  # the number of pixels in one input image
        self.n_classes = 20

        # Network Parameters
        self.n_prototypes = 15  # the number of prototypes
        self.n_layers = 4

        # height and width of each layers' filters
        self.f_1 = 3
        self.f_2 = 3
        self.f_3 = 3
        self.f_4 = 3

        # stride size in each direction for each of the layers
        self.s_1 = 2
        self.s_2 = 2
        self.s_3 = 2
        self.s_4 = 2

        # number of feature maps in each layer
        self.n_map_1 = 32
        self.n_map_2 = 32
        self.n_map_3 = 32
        # n_map_4 = 10
        self.n_map_4 = 20  # 20

        # the shapes of each layer's filter
        self.filter_shape_1 = [self.f_1, self.f_1, self.n_input_channel, self.n_map_1]
        self.filter_shape_2 = [self.f_2, self.f_2, self.n_map_1, self.n_map_2]
        self.filter_shape_3 = [self.f_3, self.f_3, self.n_map_2, self.n_map_3]
        self.filter_shape_4 = [self.f_4, self.f_4, self.n_map_3, self.n_map_4]

        self.stride_1 = [1, self.s_1, self.s_1, 1]
        self.stride_2 = [1, self.s_2, self.s_2, 1]
        self.stride_3 = [1, self.s_3, self.s_3, 1]
        self.stride_4 = [1, self.s_4, self.s_4, 1]

        if color_channels == 1:
            self.transform = {
                'data': transforms.Compose([
                    transforms.Resize(self.IMG_SIZE),
                    transforms.Grayscale(1),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: torch.flatten(x))
                ])
                }
        else:
            self.transform = {
                'data': transforms.Compose([
                    transforms.Resize(self.IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: torch.flatten(x))
                ])
            }

        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size], name='X')
        # X = tf.placeholder(dtype=tf.float32, shape=[None, input_height, input_width,n_input_channel], name='X')
        self.X_img = tf.reshape(self.X, shape=[-1, self.input_height, self.input_width, self.n_input_channel], name='X_img')
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.n_classes], name='Y')

        # We create a tf placeholder for every lambda so that they can be tweaked during training
        self.lambda_class_t = tf.placeholder(dtype=tf.float32, shape=(), name="lambda_class_t")
        self.lambda_ae_t = tf.placeholder(dtype=tf.float32, shape=(), name="lambda_ae_t")
        self.lambda_2_t = tf.placeholder(dtype=tf.float32, shape=(), name="lambda_2_t")
        self.lambda_1_t = tf.placeholder(dtype=tf.float32, shape=(), name="lambda_1_t")

        self.weights = {
            'enc_f1': tf.Variable(tf.random_normal(self.filter_shape_1,
                                                   stddev=0.01,
                                                   dtype=tf.float32),
                                  name='encoder_f1'),
            'enc_f2': tf.Variable(tf.random_normal(self.filter_shape_2,
                                                   stddev=0.01,
                                                   dtype=tf.float32),
                                  name='encoder_f2'),
            'enc_f3': tf.Variable(tf.random_normal(self.filter_shape_3,
                                                   stddev=0.01,
                                                   dtype=tf.float32),
                                  name='encoder_f3'),
            'enc_f4': tf.Variable(tf.random_normal(self.filter_shape_4,
                                                   stddev=0.01,
                                                   dtype=tf.float32),
                                  name='encoder_f4'),
            'dec_f4': tf.Variable(tf.random_normal(self.filter_shape_4,
                                                   stddev=0.01,
                                                   dtype=tf.float32),
                                  name='decoder_f4'),
            'dec_f3': tf.Variable(tf.random_normal(self.filter_shape_3,
                                                   stddev=0.01,
                                                   dtype=tf.float32),
                                  name='decoder_f3'),
            'dec_f2': tf.Variable(tf.random_normal(self.filter_shape_2,
                                                   stddev=0.01,
                                                   dtype=tf.float32),
                                  name='decoder_f2'),
            'dec_f1': tf.Variable(tf.random_normal(self.filter_shape_1,
                                                   stddev=0.01,
                                                   dtype=tf.float32),
                                  name='decoder_f1')
        }

        self.biases = {
            'enc_b1': tf.Variable(tf.zeros([self.n_map_1], dtype=tf.float32),
                                  name='encoder_b1'),
            'enc_b2': tf.Variable(tf.zeros([self.n_map_2], dtype=tf.float32),
                                  name='encoder_b2'),
            'enc_b3': tf.Variable(tf.zeros([self.n_map_3], dtype=tf.float32),
                                  name='encoder_b3'),
            'enc_b4': tf.Variable(tf.zeros([self.n_map_4], dtype=tf.float32),
                                  name='encoder_b4'),
            'dec_b4': tf.Variable(tf.zeros([self.n_map_3], dtype=tf.float32),
                                  name='decoder_b4'),
            'dec_b3': tf.Variable(tf.zeros([self.n_map_2], dtype=tf.float32),
                                  name='decoder_b3'),
            'dec_b2': tf.Variable(tf.zeros([self.n_map_1], dtype=tf.float32),
                                  name='decoder_b2'),
            'dec_b1': tf.Variable(tf.zeros([self.n_input_channel], dtype=tf.float32),
                                  name='decoder_b1')
        }

        self.last_layer = {
            'w': tf.Variable(tf.random_uniform(shape=[self.n_prototypes, self.n_classes],
                                               dtype=tf.float32),
                             name='last_layer_w')
        }
        # construct the model
        # eln means the output of the nth layer of the encoder
        self.el1 = self.conv_layer(self.X_img, self.weights['enc_f1'], self.biases['enc_b1'], self.stride_1, "SAME")
        self.el2 = self.conv_layer(self.el1, self.weights['enc_f2'], self.biases['enc_b2'], self.stride_2, "SAME")
        self.el3 = self.conv_layer(self.el2, self.weights['enc_f3'], self.biases['enc_b3'], self.stride_3, "SAME")
        self.el4 = self.conv_layer(self.el3, self.weights['enc_f4'], self.biases['enc_b4'], self.stride_4, "SAME")

        # we compute the output shape of each layer because the deconv_layer function requires it
        self.l1_shape = self.el1.get_shape().as_list()
        self.l2_shape = self.el2.get_shape().as_list()
        self.l3_shape = self.el3.get_shape().as_list()
        self.l4_shape = self.el4.get_shape().as_list()

        self.flatten_size = self.l4_shape[1] * self.l4_shape[2] * self.l4_shape[3]
        self.n_features = self.flatten_size
        # feature vectors is the flattened output of the encoder
        self.feature_vectors = tf.reshape(self.el4, shape=[-1, self.flatten_size], name='feature_vectors')

        # the list prototype feature vectors
        self.prototype_feature_vectors = tf.Variable(tf.random_uniform(shape=[self.n_prototypes, self.n_features],
                                                                  dtype=tf.float32),
                                                name='prototype_feature_vectors')

        '''deconv_batch_size is the number of feature vectors in the batch going into
        the deconvolutional network. This is required by the signature of
        conv2d_transpose. But instead of feeding in the value, the size is infered during
        sess.run by looking at how many rows the feature_vectors matrix has
        '''
        self.deconv_batch_size = tf.identity(tf.shape(self.feature_vectors)[0], name="deconv_batch_size")

        # this is necessary for prototype images evaluation
        self.reshape_feature_vectors = tf.reshape(self.feature_vectors, shape=[-1, self.l4_shape[1], self.l4_shape[2], self.l4_shape[3]])

        # dln means the output of the nth layer of the decoder
        self.dl4 = self.deconv_layer(self.reshape_feature_vectors, self.weights['dec_f4'], self.biases['dec_b4'],
                           output_shape=[self.deconv_batch_size, self.l3_shape[1], self.l3_shape[2], self.l3_shape[3]],
                           strides=self.stride_4, padding="SAME")
        self.dl3 = self.deconv_layer(self.dl4, self.weights['dec_f3'], self.biases['dec_b3'],
                           output_shape=[self.deconv_batch_size, self.l2_shape[1], self.l2_shape[2], self.l2_shape[3]],
                           strides=self.stride_3, padding="SAME")
        self.dl2 = self.deconv_layer(self.dl3, self.weights['dec_f2'], self.biases['dec_b2'],
                           output_shape=[self.deconv_batch_size, self.l1_shape[1], self.l1_shape[2], self.l1_shape[3]],
                           strides=self.stride_2, padding="SAME")
        self.dl1 = self.deconv_layer(self.dl2, self.weights['dec_f1'], self.biases['dec_b1'],
                           output_shape=[self.deconv_batch_size, self.input_height, self.input_width, self.n_input_channel],
                           strides=self.stride_1, padding="SAME", nonlinearity=tf.nn.sigmoid)
        '''
        X_decoded is the decoding of the encoded feature vectors in X;
        we reshape it to match the shape of the training input
        X_true is the correct output for the autoencoder
        '''
        self.X_decoded = tf.reshape(self.dl1, shape=[-1, self.input_size], name='X_decoded')
        # X_decoded = tf.reshape(dl1, shape=[-1, input_height, input_width,n_input_channel], name='X_decoded')
        self.X_true = tf.identity(self.X, name='X_true')
        self.history = {
            "train": [],
            "valid": [],
            "test": []
        }

    def conv_layer(self,input, filter, bias, strides, padding="VALID", nonlinearity=tf.nn.relu):
        conv = tf.nn.conv2d(input, filter, strides=strides, padding=padding)
        act = nonlinearity(conv + bias)
        return act

    # tensorflow's conv2d_transpose needs to know the shape of the output
    def deconv_layer(self,input, filter, bias, output_shape, strides, padding="VALID", nonlinearity=tf.nn.relu):
        deconv = tf.nn.conv2d_transpose(input, filter, output_shape, strides, padding=padding)
        act = nonlinearity(deconv + bias)
        return act

    def fc_layer(self,input, weight, bias, nonlinearity=tf.nn.relu):
        return nonlinearity(tf.matmul(input, weight) + bias)

    def prepare_data(self):
        dataset = CustomImageFolder(root=self.PATH, transform=self.transform['data'])
        classes = dataset.class_to_idx.keys()
        test = dataset.class_to_idx

        test_size = int(0.1 * len(dataset))
        valid_size = int(0.3 * len(dataset))

        # making sure we shuffle the animals around so it learns something
        all_indices = set(range(len(dataset)))
        team1 = set(random.sample(all_indices, int(len(dataset) / 2)))
        team2 = all_indices - team1
        team1 = sorted(team1, key=lambda x: random.random())
        team2 = sorted(team2, key=lambda x: random.random())

        # getting the indices for each dataset
        training = team1
        validating = team2[:int(len(team2) / 2)]
        testing = team2[int(len(team2) / 2):]

        # take the proper subsets
        test_set = torch.utils.data.Subset(dataset, testing)  # take 10% for test
        valid_set = torch.utils.data.Subset(dataset, validating)
        train_set = torch.utils.data.Subset(dataset, training)

        # create a list with images for later displaying
        data = []
        for i in test_set:
            data.append(i[0])
        data = torch.stack(data)

        # create DataLoader object from the sets created above given batch size
        train = DataLoader(train_set, shuffle=True, batch_size=self.batch_size)
        valid = DataLoader(valid_set, shuffle=True, batch_size=self.batch_size)
        test = DataLoader(test_set, shuffle=True, batch_size=self.batch_size)

        return train_set, valid_set, test_set, data, train, valid, test, classes

    def run(self):
        train_set, valid_set, test_set, data, train, valid, test, classes = self.prepare_data()
        makedirs(self.model_folder)
        makedirs(self.prototype_folder)
        img_folder = os.path.join(self.model_folder, "img")
        makedirs(img_folder)
        console_log = open(os.path.join(self.model_folder, "console_log.txt"), "w+")

        prototype_distances = list_of_distances(self.feature_vectors,
                                                self.prototype_feature_vectors)
        prototype_distances = tf.identity(prototype_distances, name='prototype_distances')
        feature_vector_distances = list_of_distances(self.prototype_feature_vectors,
                                                     self.feature_vectors)
        feature_vector_distances = tf.identity(feature_vector_distances, name='feature_vector_distances')

        # the logits are the weighted sum of distances from prototype_distances
        logits = tf.matmul(prototype_distances, self.last_layer['w'], name='logits')
        probability_distribution = tf.nn.softmax(logits=logits,
                                                 name='probability_distribution')

        '''
        the error function consists of 4 terms, the autoencoder loss,
        the classification loss, and the two requirements that every feature vector in
        X look like at least one of the prototype feature vectors and every prototype
        feature vector look like at least one of the feature vectors in X.
        '''
        ae_error = tf.reduce_mean(list_of_norms(self.X_decoded - self.X_true), name='ae_error')
        class_error = tf.losses.softmax_cross_entropy(onehot_labels=self.Y, logits=logits)
        class_error = tf.identity(class_error, name='class_error')
        error_1 = tf.reduce_mean(tf.reduce_min(feature_vector_distances, axis=1), name='error_1')
        error_2 = tf.reduce_mean(tf.reduce_min(prototype_distances, axis=1), name='error_2')

        # total_error is the our minimization objective
        total_error = self.lambda_class_t * class_error + \
                      self.lambda_ae_t * ae_error + \
                      self.lambda_1_t * error_1 + \
                      self.lambda_2_t * error_2
        total_error = tf.identity(total_error, name='total_error')

        # accuracy is not the classification error term; it is the percentage accuracy
        correct_prediction = tf.equal(tf.argmax(logits, 1),
                                      tf.argmax(self.Y, 1),
                                      name='correct_prediction')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32),
                                  name='accuracy')

        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(total_error)
        # add the optimizer to collection so that we can retrieve the optimizer and resume training
        tf.add_to_collection("optimizer", optimizer)

        # Create the variable init operation and a saver object to store the model
        init = tf.global_variables_initializer()

        hyperparameters = {
            "learning_rate": self.learning_rate,
            "training_epochs": self.training_epochs,
            "batch_size": self.batch_size,
            "test_display_step": self.test_display_step,
            "save_step": self.save_step,

            "lambda_class": self.lambda_class,
            "lambda_ae": self.lambda_ae,
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,

            "input_height": self.input_height,
            "input_width": self.input_width,
            "n_input_channel": self.n_input_channel,
            "input_size": self.input_size,
            "n_classes": self.n_classes,

            "n_prototypes": self.n_prototypes,
            "n_layers": self.n_layers,

            "f_1": self.f_1,
            "f_2": self.f_2,
            "f_3": self.f_3,
            "f_4": self.f_4,

            "s_1": self.s_1,
            "s_2": self.s_2,
            "s_3": self.s_3,
            "s_4": self.s_4,

            "n_map_1": self.n_map_1,
            "n_map_2": self.n_map_2,
            "n_map_3": self.n_map_3,
            "n_map_4": self.n_map_4,

            "n_features": self.n_features,
        }
        # save the hyperparameters above in the model snapshot
        for (name, value) in hyperparameters.items():
            tf.add_to_collection('hyperparameters', tf.constant(name=name, value=value))

        saver = tf.train.Saver(max_to_keep=self.n_saves)
        last_ep = 0
        config = tf.ConfigProto()
        # the amount of GPU memory our process occupies
        config.gpu_options.per_process_gpu_memory_fraction = 0.3

        with tf.Session(config=config) as sess:
            sess.run(init)
            # we compute the number of batches because both training and evaluation
            # happens batch by batch; we do not throw the entire test set onto the GPU
            # n_train_batch = mnist.train.num_examples // batch_size
            # n_valid_batch = mnist.validation.num_examples // batch_size
            # n_test_batch = mnist.test.num_examples // batch_size

            n_train_batch = len(train_set) // self.batch_size
            n_valid_batch = len(valid_set) // self.batch_size
            n_test_batch = len(test_set) // self.batch_size
            print(n_train_batch, n_valid_batch, n_test_batch)
            # Training cycle
            for epoch in range(self.training_epochs):
                print_and_write("#" * 80, console_log)
                print_and_write("Epoch: %04d" % (epoch), console_log)
                start_time = time.time()
                train_ce, train_ae, train_e1, train_e2, train_te, train_ac = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                # Loop over all batches
                for i in range(n_train_batch):
                    # batch_x, batch_y = mnist.train.next_batch(batch_size)
                    batch_x, batch_y = next(iter(train))
                    if self.color_channels==1:
                        elastic_batch_x = batch_elastic_transform(batch_x, sigma=self.sigma, alpha=self.alpha, height=self.input_height,
                                                              width=self.input_width)
                    else:
                        elastic_batch_x = batch_elastic_transform_color(batch_x, sigma=self.sigma, alpha=self.alpha,
                                                                  height=self.input_height,
                                                                  width=self.input_width)
                    # elastic_batch_x = elastic_batch_x.reshape((32, input_width, input_width,3))
                    _, ce, ae, e1, e2, te, ac = sess.run(
                        (optimizer,
                         class_error,
                         ae_error,
                         error_1,
                         error_2,
                         total_error,
                         accuracy),
                        feed_dict={self.X: elastic_batch_x,
                                   self.Y: batch_y,
                                   self.lambda_class_t: self.lambda_class,
                                   self.lambda_ae_t: self.lambda_ae,
                                   self.lambda_1_t: self.lambda_1,
                                   self.lambda_2_t: self.lambda_2})
                    train_ce += (ce / n_train_batch)
                    train_ae += (ae / n_train_batch)
                    train_e1 += (e1 / n_train_batch)
                    train_e2 += (e2 / n_train_batch)
                    train_te += (te / n_train_batch)
                    train_ac += (ac / n_train_batch)
                self.history["train"].append(train_ac)
                end_time = time.time()
                print_and_write('training takes {0:.2f} seconds.'.format((end_time - start_time)), console_log)
                # after every epoch, check the error terms on the entire training set
                print_and_write("training set errors:", console_log)
                print_and_write("\tclassification error: {:.6f}".format(train_ce), console_log)
                print_and_write("\tautoencoder error: {:.6f}".format(train_ae), console_log)
                print_and_write("\terror_1: {:.6f}".format(train_e1), console_log)
                print_and_write("\terror_2: {:.6f}".format(train_e2), console_log)
                print_and_write("\ttotal error: {:.6f}".format(train_te), console_log)
                print_and_write("\taccuracy: {:.4f}".format(train_ac), console_log)

                # validation set error terms evaluation
                valid_ce, valid_ae, valid_e1, valid_e2, valid_te, valid_ac = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                # Loop over all batches
                for i in range(n_valid_batch):
                    batch_x, batch_y = next(iter(valid))
                    # batch_x = batch_x.reshape(32, input_width, input_height, n_input_channel)
                    ce, ae, e1, e2, te, ac = sess.run(
                        (class_error,
                         ae_error,
                         error_1,
                         error_2,
                         total_error,
                         accuracy),
                        feed_dict={self.X: batch_x,
                                   self.Y: batch_y,
                                   self.lambda_class_t: self.lambda_class,
                                   self.lambda_ae_t: self.lambda_ae,
                                   self.lambda_2_t: self.lambda_2,
                                   self.lambda_1_t: self.lambda_1})
                    valid_ce += ce / n_valid_batch
                    valid_ae += ae / n_valid_batch
                    valid_e1 += e1 / n_valid_batch
                    valid_e2 += e2 / n_valid_batch
                    valid_te += te / n_valid_batch
                    valid_ac += ac / n_valid_batch
                self.history["valid"].append(valid_ac)
                # after every epoch, check the error terms on the entire training set
                print_and_write("validation set errors:", console_log)
                print_and_write("\tclassification error: {:.6f}".format(valid_ce), console_log)
                print_and_write("\tautoencoder error: {:.6f}".format(valid_ae), console_log)
                print_and_write("\terror_1: {:.6f}".format(valid_e1), console_log)
                print_and_write("\terror_2: {:.6f}".format(valid_e2), console_log)
                print_and_write("\ttotal error: {:.6f}".format(valid_te), console_log)
                print_and_write("\taccuracy: {:.4f}".format(valid_ac), console_log)

                # test set accuracy evaluation
                if epoch % self.test_display_step == 0 or epoch == self.training_epochs - 1:
                    test_ac = 0.0
                    for i in range(n_test_batch):
                        batch_x, batch_y = next(iter(test))
                        # batch_x = batch_x.reshape(32, input_width, input_height, n_input_channel)
                        ac = sess.run(accuracy,
                                      feed_dict={self.X: batch_x,
                                                 self.Y: batch_y})
                        test_ac += ac / n_test_batch
                    self.history["test"].append(test_ac)
                    # after every epoch, check the error terms on the entire training set
                    print_and_write("test set:", console_log)
                    print_and_write("\taccuracy: {:.4f}".format(test_ac), console_log)

                if epoch % self.save_step == 0 or epoch == self.training_epochs - 1:
                    # one .meta file is enough to recover the computational graph
                    saver.save(sess, os.path.join(self.model_folder, self.model_filename),
                               global_step=epoch,
                               write_meta_graph=(epoch == 0 or epoch == self.training_epochs - 1))
                    prototype_imgs = sess.run(self.X_decoded,
                                              feed_dict={self.feature_vectors: self.prototype_feature_vectors.eval()})
                    # visualize the prototype images
                    n_cols = 5
                    n_rows = self.n_prototypes // n_cols + 1 if self.n_prototypes % n_cols != 0 else self.n_prototypes // n_cols
                    g, b = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
                    # g, b = plt.subplots(n_rows, n_cols, figsize=(10, 10))
                    for i in range(n_rows):
                        for j in range(n_cols):
                            if i * n_cols + j < self.n_prototypes:
                                if self.color_channels==1:
                                    b[i][j].imshow(
                                        torch.from_numpy(prototype_imgs[i * n_cols + j]).reshape(self.input_height, self.input_width),
                                        cmap='gray',
                                        interpolation='none')
                                    b[i][j].axis('off')
                                else:
                                    b[i][j].imshow(
                                        torch.from_numpy(prototype_imgs[i * n_cols + j]).reshape(3, self.input_height,
                                                                                                 self.input_width).permute(1,
                                                                                                                      2,
                                                                                                                      0),
                                        cmap='jet',
                                        interpolation='none')
                                    b[i][j].axis('off')

                    plt.savefig(os.path.join(img_folder, 'prototype_result-' + str(epoch) + '.png'),
                                transparent=True,
                                bbox_inches='tight',
                                pad_inches=0)
                    plt.close()

                    # generating single images
                    for i in range(n_rows):
                        for j in range(n_cols):
                            if i * n_cols + j < self.n_prototypes:
                                l, h = plt.subplots(1, 1, figsize=(1, 1))
                                if self.color_channels==1:
                                    h.imshow(
                                        torch.from_numpy(prototype_imgs[i * n_cols + j]).reshape(self.input_height, self.input_width),
                                        cmap='gray',
                                        interpolation='none')
                                    h.axis('off')
                                else:
                                    h.imshow(torch.from_numpy(prototype_imgs[i * n_cols + j]).reshape(3, self.input_height,
                                                                                                      self.input_width).permute(
                                        1, 2, 0),
                                             cmap='jet',
                                             interpolation='none')
                                    h.axis('off')

                                plt.savefig(
                                    os.path.join(self.prototype_folder, 'prototype_result-' + str(i) + str(j) + '.png'),
                                    transparent=True,
                                    bbox_inches='tight',
                                    pad_inches=0)
                                plt.close()

                    # Applying encoding and decoding over a small subset of the training set
                    examples_to_show = 10
                    #            encode_decode = sess.run(X_decoded,
                    #                                     feed_dict={X: mnist.train.images[:examples_to_show]})
                    encode_decode = sess.run(self.X_decoded,
                                             feed_dict={self.X: data[:examples_to_show]})
                    # Compare original images with their reconstructions
                    f, a = plt.subplots(2, examples_to_show, figsize=(examples_to_show, 2))
                    for i in range(examples_to_show):
                        if self.color_channels==1:
                            a[0][i].imshow(data[i].reshape(self.input_height, self.input_width),
                                       cmap='gray',
                                       interpolation='none')
                            a[0][i].axis('off')
                            a[1][i].imshow(torch.from_numpy(encode_decode[i]).reshape(self.input_height, self.input_width),
                                       cmap='gray',
                                       interpolation='none')
                            a[1][i].axis('off')
                        else:
                            a[0][i].imshow(data[i].reshape(3, self.input_height, self.input_width).permute(1, 2, 0),
                                           cmap='jet',
                                           interpolation='none')
                            a[0][i].axis('off')
                            # a[1][i].imshow(encode_decode[i].reshape(input_height, input_width, n_input_channel),
                            #               cmap='jet',
                            #               interpolation='none')
                            a[1][i].imshow(
                                torch.from_numpy(encode_decode[i]).reshape(3, self.input_height, self.input_width).permute(1, 2,
                                                                                                                 0),
                                cmap='jet',
                                interpolation='none')
                            a[1][i].axis('off')

                    plt.savefig(os.path.join(img_folder, 'decoding_result-' + str(epoch) + '.png'),
                                transparent=True,
                                bbox_inches='tight',
                                pad_inches=0)
                    plt.close()
            print_and_write("Optimization Finished!", console_log)
            last_ep = epoch
        console_log.close()
        self.save_matrix(classes, last_ep)
    def plot(self):
        plt.plot(self.history['train'])
        plt.plot(self.history['valid'])
        plt.plot(self.history['test'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val', 'test'], loc='upper left')
        plt.savefig(f'{self.model_folder}/accuracies.png')
    def save_matrix(self, classes,last_epoch):
        path_to_pics = f"{self.model_folder}/prototypes"
        reader = tf.train.load_checkpoint(
            f'{self.model_folder}/mnist_cae-{last_epoch}')
        shape_from_key = reader.get_variable_to_shape_map()
        dtype_from_key = reader.get_variable_to_dtype_map()

        cols = classes
        values = reader.get_tensor('last_layer_w')
        df = pd.DataFrame(values, columns=cols)

        img_paths = []
        for images in glob.iglob(f'{path_to_pics}/*'):

            # check if the image ends with png
            if (images.endswith(".png")):
                img_paths.append(images)
        img_paths = sorted(img_paths)

        df['image'] = img_paths
        df['image'] = '''<img src="''' + df['image'] + '''">'''
        # df.style.applymap(_color_red_or_green)
        df.style.background_gradient(cmap="coolwarm").to_excel(
            f"{self.prototype_folder}/table.xlsx")
        # df = df.style.applymap(_color_red_or_green)
        with open(
                f'{self.prototype_folder}/matrix.html',
                'w') as fo:
            fo.write(df.to_html(render_links=True, escape=False))


class PreprocessData():
    @staticmethod
    def preprocess_data(path, flag):
        if flag==1:
            df = pd.read_csv(path)

            pixels = df.iloc[:, 1:].values
            labels = df['label'].values

            # Reshape the pixel values to a 28x28 image (assuming the images are 28x28 pixels)
            images = pixels.reshape(-1, 28, 28)
            path_to_save = ""
            for image, label in zip(images, labels):
                num = random.random()
                folder_to_save = os.path.join(os.getcwd(), "sign_lang", "data", f"{label}")
                # makedirs(folder_to_save)
                path_to_save = os.path.join(os.getcwd(), "sign_lang", "data", f"{label}")
                makedirs(path_to_save)
                im = PIL.Image.fromarray((image * 255).astype(np.uint8))
                im.save(f"{path_to_save}/{num}.jpg")
            return path_to_save
        else:
            return path
class CustomImageFolder(ImageFolder):
    """
    Please note that the structure of the image folder must folow the following format
    directory/
├── class_x
│   ├── xxx.ext
│   ├── xxy.ext
│   └── ...
│       └── xxz.ext
└── class_y
    ├── 123.ext
    ├── nsdf3.ext
    └── ...
    └── asd932_.ext

    If your image dataset is not in this structure, you need to preprocess it. You can inject your function for
    preprocessing in the class PreprocessData. There are already some options inside so please first check if
    your structure follows any of the already written.
    """
    def get_labels(self, path):
        directory_list = list()
        for root, dirs, files in os.walk(path, topdown=False):
            for name in dirs:
                directory_list.append(name)
        return directory_list
    def indices_to_one_hot(self,nb_classes):
        data = [list(range(0, nb_classes))]
        """Convert an iterable of indices to one-hot encoded labels."""
        targets = np.array(data).reshape(-1)
        return np.eye(nb_classes)[targets]
    def classes_to_idx(self):
        animals = self.get_labels(self.root)
        one_hot_encoded = self.indices_to_one_hot(len(animals))
        res = {}
        for i,n in enumerate(animals):
            res[n] = one_hot_encoded[i]
        return res
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Override this method to load from setting file instead of scanning directory
        """
        classes = list(self.classes_to_idx().keys())
        classes_to_idx = self.classes_to_idx()
        return classes, classes_to_idx

if __name__ == "__main__":
    import argparse
    import os
    import glob

    # DO NOT CHANGE THIS!
    parser = argparse.ArgumentParser(description="Meter Estimation")
    parser.add_argument(
        "--datadir",
        "-i",
        help="path to the input files",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--channels",
        "-c",
        help="color channels",
        default=False,
        type=int
    )

    parser.add_argument(
        "--height",
        "-he",
        help="height and width of the images",
        type=int,
        default="meter_estimation.txt",
    )

    parser.add_argument(
        "--model-folder",
        "-mf",
        help="model folder where images are saved",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--flag",
        "-f",
        help="model folder where images are saved",
        type=int,
        default=None,
    )

    args = parser.parse_args()
    model = PrototypeDL(args.datadir, args.channels, args.height, args.model_folder, args.flag)
    model.run()


