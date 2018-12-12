# Gade, Aniket
# 1001-505-046
# 2018-12-09
# Assignment-06-01
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 23:39:11 2018

@author: Aniket Gade
"""

import sys
import os
import sklearn.datasets
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tensorflow as tf
if sys.version_info[0] < 3:
    from Tkinter import *
else:
    from tkinter import *
import Gade_06_02

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    # Creation of init_window
    def init_window(self):
        # initialization
        self.learning_rate = 0.1
        self.regularization_rate = 0.01
        self.training_percent = 20
        self.f1 = 32
        self.k1 = 3
        self.f2 = 32
        self.k2 = 3
        self.loss_list = [-1]
        self.class_names = None
        self.images_train = None
        self.labels_train = None
        self.images_test = None
        self.labels_test = None
        self.cost = None
        self.optimizer = None
        self.first_init = True
        # generate data
        path = os.getcwd()
        path = path + "/Data/"
        Gade_06_02.data_path = path
        Gade_06_02.maybe_download_and_extract()
        self.class_names = Gade_06_02.load_class_names()
        self.images_train, _, self.labels_train = Gade_06_02.load_training_data()
        self.images_test, _, self.labels_test = Gade_06_02.load_test_data()        
        
        self.images_train = np.array(self.images_train[:self.training_percent*500], dtype="float32")
        self.labels_train = self.labels_train[:self.training_percent*500]       
        
        self.images_test = np.array(self.images_test, dtype="float32")
        
        self.weights = {'W_conv1': tf.Variable(tf.random_normal([self.k1, self.k1, 3, self.f1], stddev=0.1)),
                         'W_conv2': tf.Variable(tf.random_normal([self.k2, self.k2, self.f1, self.f2], stddev=0.1)),
                         'W_conv3': tf.Variable(tf.random_normal([3, 3, self.f2, 32], stddev=0.1)),
                         'W_fc': tf.Variable(tf.random_normal([512, 10], stddev=0.1))}
        
        self.biases = {'b_conv1': tf.Variable(tf.random_normal([self.f1], stddev=0.08)),
                       'b_conv2': tf.Variable(tf.random_normal([self.f2], stddev=0.08)),
                       'b_conv3': tf.Variable(tf.random_normal([32], stddev=0.08)),
                       'b_fc': tf.Variable(tf.random_normal([10], stddev=0.08))}
        
        self.x = tf.placeholder('float32',[None, 32, 32, 3])
        self.y = tf.placeholder('float32',[None, 10])
        
        init_operator = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_operator)
        
        self.init_model()
        print("Model Initialized!")
#        init_operator = tf.global_variables_initializer()
#        self.sess = tf.Session()
#        self.sess.run(init_operator)
              
        
        # changing the title of our master widget
        self.master.title("TensorFlow Back propagation")

        self.left_frame = Frame(self.master)
        self.right_frame = Frame(self.master)
        self.conrols = Frame(self.master)

        self.left_frame.grid(row=0, column=0)
        self.right_frame.grid(row=0, column=1)
        self.conrols.grid(row=1, columnspan=2)

        self.figure = plt.figure(figsize=(5, 5))
        # self.axes = self.figure.add_axes([0.15, 0.15, 0.80, 0.80])
        self.axes = self.figure.add_axes()
        self.axes = self.figure.gca()
        self.axes.set_title("")
        self.axes.set_aspect('auto')
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.left_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0)

        self.learning_rate_slider = Scale(self.conrols, variable=DoubleVar(), orient=HORIZONTAL,
                                          from_=0.000, to_=1, resolution=0.001, bg="#DDDDDD",
                                          activebackground="#FF0000", highlightcolor="#00FFFF",
                                          label="Learning Rate(Alpha)",
                                          command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=0)

        self.weight_regularization_slider = Scale(self.conrols, variable=DoubleVar(), orient=HORIZONTAL,
                                                  from_=0, to_=1, resolution=0.01, bg="#DDDDDD",
                                                  activebackground="#FF0000", highlightcolor="#00FFFF",
                                                  label="Lambda",
                                                  command=lambda event: self.weight_regularization_slider_callback())
        self.weight_regularization_slider.set(self.regularization_rate)
        self.weight_regularization_slider.bind("<ButtonRelease-1>",
                                               lambda event: self.weight_regularization_slider_callback())
        self.weight_regularization_slider.grid(row=0, column=1)

        self.f1_slider = Scale(self.conrols, variable=DoubleVar(), orient=HORIZONTAL,
                                                     from_=1, to_=64, resolution=1, bg="#DDDDDD",
                                                     activebackground="#FF0000", highlightcolor="#00FFFF",
                                                     label="F1",
                                                     command=lambda
                                                     event: self.f1_slider_callback())
        self.f1_slider.set(self.f1)
        self.f1_slider.bind("<ButtonRelease-1>",lambda event: self.f1_slider_callback())
        self.f1_slider.grid(row=0, column=2)
        
        
        self.f2_slider = Scale(self.conrols, variable=DoubleVar(), orient=HORIZONTAL,
                                                     from_=1, to_=64, resolution=1, bg="#DDDDDD",
                                                     activebackground="#FF0000", highlightcolor="#00FFFF",
                                                     label="F2",
                                                     command=lambda
                                                     event: self.f2_slider_callback())
        self.f2_slider.set(self.f2)
        self.f2_slider.bind("<ButtonRelease-1>",lambda event: self.f2_slider_callback())
        self.f2_slider.grid(row=0, column=3)

        self.k1_slider = Scale(self.conrols, variable=DoubleVar(), orient=HORIZONTAL,
                                                     from_=3, to_=7, resolution=1, bg="#DDDDDD",
                                                     activebackground="#FF0000", highlightcolor="#00FFFF",
                                                     label="K1",
                                                     command=lambda
                                                     event: self.k1_slider_callback())
        self.k1_slider.set(self.k1)
        self.k1_slider.bind("<ButtonRelease-1>",lambda event: self.k1_slider_callback())
        self.k1_slider.grid(row=0, column=4)


        self.k2_slider = Scale(self.conrols, variable=DoubleVar(), orient=HORIZONTAL,
                                                     from_=3, to_=7, resolution=1, bg="#DDDDDD",
                                                     activebackground="#FF0000", highlightcolor="#00FFFF",
                                                     label="K2",
                                                     command=lambda
                                                     event: self.k2_slider_callback())
        self.k2_slider.set(self.k2)
        self.k2_slider.bind("<ButtonRelease-1>",lambda event: self.k2_slider_callback())
        self.k2_slider.grid(row=0, column=5)

        self.train_precent_slider = Scale(self.conrols, variable=DoubleVar(), orient=HORIZONTAL,
                                                     from_=1, to_=100, resolution=1, bg="#DDDDDD",
                                                     activebackground="#FF0000", highlightcolor="#00FFFF",
                                                     label="Training Set %",
                                                     command=lambda
                                                     event: self.train_percent_slider_callback())
        self.train_precent_slider.set(self.training_percent)
        self.train_precent_slider.bind("<ButtonRelease-1>",lambda event: self.train_percent_slider_callback())
        self.train_precent_slider.grid(row=0, column=6)

        self.adjust_weights = Button(self.conrols, text="Adjust Weights(Train)", command=self.train_model)
        self.adjust_weights.grid(row=1, column=0)

        self.reset_weights = Button(self.conrols, text="Reset Weights", command=self.reset_weights_fun)
        self.reset_weights.grid(row=1, column=1)
        
        self.update_parameters = Button(self.conrols, text="Update Parameters", command=self.update_parameters)
        self.update_parameters.grid(row=2, column=0)

    def reset_weights_fun(self):
        tf.reset_default_graph()
        self.sess.close()
        # tensorflow part
        self.weights = {'W_conv1': tf.Variable(tf.random_normal([self.k1, self.k1, 3, self.f1], stddev=0.1)),
                         'W_conv2': tf.Variable(tf.random_normal([self.k2, self.k2, self.f1, self.f2], stddev=0.1)),
                         'W_conv3': tf.Variable(tf.random_normal([3, 3, self.f2, 32], stddev=0.1)),
                         'W_fc': tf.Variable(tf.random_normal([512, 10], stddev=0.1))}
        
        self.biases = {'b_conv1': tf.Variable(tf.random_normal([self.f1], stddev=0.08)),
                       'b_conv2': tf.Variable(tf.random_normal([self.f2], stddev=0.08)),
                       'b_conv3': tf.Variable(tf.random_normal([32], stddev=0.08)),
                       'b_fc': tf.Variable(tf.random_normal([10], stddev=0.08))}
        
        self.x = tf.placeholder('float32',[None, 32, 32, 3])
        self.y = tf.placeholder('float32',[None, 10])

        init_operator = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_operator)
        self.init_model()
        self.train_model()

    def learning_rate_slider_callback(self):
        self.learning_rate = np.float(self.learning_rate_slider.get())
        print(self.learning_rate)

    def weight_regularization_slider_callback(self):
        self.regularization_rate = np.float(self.weight_regularization_slider.get())
        print(self.regularization_rate)

    def f1_slider_callback(self):
        self.f1 = np.int(self.f1_slider.get())
        print(self.f1)
        
    def f2_slider_callback(self):
        self.f2 = np.int(self.f2_slider.get())
        print(self.f2)
        
    def k1_slider_callback(self):
        self.k1 = np.int(self.k1_slider.get())
        print(self.k1)
        
    def k2_slider_callback(self):
        self.k2 = np.int(self.k2_slider.get())
        print(self.k2)
        
    def train_percent_slider_callback(self):
        self.training_percent = np.int(self.train_precent_slider.get())
        print(self.training_percent)
        
    def update_parameters(self):
        path = os.getcwd()
        path = path + "/Data/"
        Gade_06_02.data_path = path
        Gade_06_02.maybe_download_and_extract()
        self.class_names = Gade_06_02.load_class_names()
        self.images_train, _, self.labels_train = Gade_06_02.load_training_data()
        self.images_test, _, self.labels_test = Gade_06_02.load_test_data()        
        
        self.images_train = np.array(self.images_train[:self.training_percent*500], dtype="float32")
        self.labels_train = self.labels_train[:self.training_percent*500]
        
        self.images_test = np.array(self.images_test, dtype="float32")
        
        print("Alpha = ", self.learning_rate, " Lambda = ", self.regularization_rate, " F1, F2, K1, K2 = ", self.f1, self.f2, self.k1, self.k2)
        print(" Train Images = ", len(self.images_train), "Train Labels = ", len(self.labels_train))
        print(" Test Images = ", len(self.images_test), "Test Labels = ", len(self.labels_test))
        self.init_model()
   
    def nn_model(self, cf_flag):   
        if (cf_flag):
            images = self.images_test
            images = images.reshape((len(images), 3, 32, 32)).transpose(0, 2, 3, 1)
            
        else:
            images = self.images_train
            images = images.reshape((len(images), 3, 32, 32)).transpose(0, 2, 3, 1)
        
        conv1 = tf.nn.conv2d(images, self.weights['W_conv1'], strides = [1,1,1,1], padding = 'SAME') + self.biases['b_conv1']
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
        
        conv2 = tf.nn.conv2d(conv1, self.weights['W_conv2'], strides = [1,1,1,1], padding = 'SAME') + self.biases['b_conv2']
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
        
        conv3 = tf.nn.conv2d(conv2, self.weights['W_conv3'], strides = [1,1,1,1], padding = 'SAME') + self.biases['b_conv3']
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.nn.max_pool(conv3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
        
        flat = tf.contrib.layers.flatten(conv3)
#        fc_layer = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=10, activation_fn=None)
        fc_layer = tf.matmul(flat, self.weights['W_fc']) + self.biases['b_fc']
        return fc_layer  

    def train_model(self):
        print("Training!")
        self.prediction = self.nn_model(cf_flag = False)
        _, epoch_loss = self.sess.run([self.optimizer, self.cost], feed_dict = {self.x:self.images_train, self.y:self.labels_train})
        regularizers = tf.nn.l2_loss(self.weights['W_conv1']) + tf.nn.l2_loss(self.weights['W_conv2']) + tf.nn.l2_loss(self.weights['W_conv3']) + tf.nn.l2_loss(self.weights['W_fc'])
        cost = tf.reduce_mean(epoch_loss + self.regularization_rate * regularizers)
        loss = cost.eval(session=self.sess)
        print("Epoch Loss = ", loss)
        self.loss_list.append(loss)
        self.axes.plot(self.loss_list)
        self.canvas.draw()
        correct = tf.equal( tf.arg_max(self.prediction, 1), tf.arg_max(self.labels_train, 1))   
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print("Accuracy = ", accuracy.eval(session=self.sess, feed_dict={self.x:self.images_train, self.y:self.labels_train}))
        self.get_cf_matrix()


    def init_model(self):
        if (self.first_init):
            self.get_cf_matrix()
        else:
            self.prediction = self.nn_model(cf_flag = False)
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.labels_train))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.cost)
    #        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.cost)
        

    def get_cf_matrix(self):
        predicted = self.nn_model(cf_flag = True)
        y_pred = tf.argmax(predicted, axis = 1)
        y_true = tf.argmax(self.labels_test, axis = 1)
        confusion_matrix = tf.confusion_matrix(labels = y_true, predictions = y_pred, num_classes = 10)
        print("\n Confusion Matrix for Testing Data:")
        print(confusion_matrix.eval(session=self.sess))
        if(self.first_init):
            self.first_init = False
            self.init_model()
        
                

if __name__ == "__main__":
    root = Tk()
    # size of the window
    root.geometry("1200x650")
    app = Window(root)
    root.mainloop()