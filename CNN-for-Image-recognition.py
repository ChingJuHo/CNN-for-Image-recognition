# -*- coding: utf-8 -*-

import os
import time
import math
from PIL import Image
import numpy as np
import tensorflow as tf
tf.reset_default_graph()
tf.InteractiveSession.close
import matplotlib.pyplot as plt
from matplotlib import rc
plt.close('all')

#%% functions
def loadImages(path, folder1, folder2, width, height):
    images = []
    filenames = []
    folderPath = path + folder1 + '/' + folder2+'/'
    for filename in os.listdir(folderPath):
        try:
            img = Image.open(os.path.join(folderPath, filename))
            if img is not None:
                if img.mode == 'RGBA':
                    img.load()
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
                    img = background
                if img.mode == 'CMYK':
                    img = img.convert('RGB')
                if img.mode == 'L':
                    img = img.convert('RGB')
#                img = img.convert('L')
                img = img.resize((width,height))    #resizing
                img = np.asarray(img).reshape((width,height,3))
                images.append(img)
                filenames.append(filename)
        except:
            print('Cannot import ' + filename)
    images = np.asarray(images)
    return images, filenames

# construct a convolution layer
def addConvLayer(inputs, filter_num, kernel_size, conv_name, activation_function=tf.nn.relu):
    conv = tf.layers.Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', name=conv_name, 
                            kernel_initializer=tf.orthogonal_initializer()).apply(inputs)
    conv = tf.layers.BatchNormalization(name=conv_name+'bn').apply(conv)
    if activation_function is None:
        outputs = conv
    else:
        outputs = activation_function(conv)
    outputs = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid', name=conv_name+'pool').apply(outputs)
    return outputs

# construct a fully-connected layer
def addFcLayer(inputs, in_size, out_size, Wname, Bname, drop_rate, activation_function=tf.nn.relu):
    Weights = tf.get_variable(Wname, [in_size, out_size], initializer = tf.orthogonal_initializer())
    biases = tf.get_variable(Bname, [out_size], initializer = tf.zeros_initializer())
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, rate = drop_rate)   # Dropout
    Wx_plus_b = tf.layers.BatchNormalization(name=Wname+'bn').apply(Wx_plus_b)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# change the order of the training data at every epoch
def randomShuffle(X_, Y_):
    # create a random index
    num_train_data = len(Y_)
    idx = np.random.choice(num_train_data, size=num_train_data, replace=False)
    # use the random index to select random inputs and labels
    x_r = X_[idx,:]
    y_r = Y_[idx]
    return x_r, y_r

# plot setting
def plotType(xlabel, ylabel, title):
    plt.xlabel(xlabel, fontsize=30)
    plt.ylabel(ylabel, fontsize=30)
    plt.title(title, fontsize=30)
    plt.legend(fontsize=20, loc='upper right')
    plt.grid(True)
    rc('xtick', labelsize=20)
    rc('ytick', labelsize=20)
    
# select examples in test to plot
def selectExample(Pred_test, True_test, val_filenameList, targetList):
    right_idx = np.random.choice(np.nonzero(Pred_test==True_test)[0], size=1, replace=False)[0]
    right_fig = val_filenameList[right_idx]
    right_label = targetList[True_test[right_idx]]
    right_pred = targetList[Pred_test[right_idx]]
    wrong_idx = np.random.choice(np.nonzero(Pred_test!=True_test)[0], size=1, replace=False)[0]
    wrong_fig = val_filenameList[wrong_idx]
    wrong_label = targetList[True_test[wrong_idx]]
    wrong_pred = targetList[Pred_test[wrong_idx]]
    return [right_idx, right_fig, right_label, right_pred], [wrong_idx, wrong_fig, wrong_label, wrong_pred]

def listAccuracy(Pred_test, True_test):
    correct = Pred_test==True_test
    n = True_test.shape[0]//10
    A = []
    for i in range(10):
        a = int(round(100*np.mean(correct[i*n:(i+1)*n])))
        A.append(a)
    return A

#%% main
if __name__ == '__main__':
#%% # load data
    start_time = time.time()
    path = './animal-10/'
    width, height = 50, 50
#    images = loadImages(path, 'val', 'elephant')
    for folder1 in os.listdir(path):
        print('Folder: ', folder1)
        targetList = os.listdir(path + folder1 + '/')
        vars()[folder1+'_filenameList'] = []
        vars()[folder1+'_input'] = []
        vars()[folder1+'_target'] = []
        for i in range(len(targetList)):
            print('LoadingClass: ', targetList[i])
            images, filenames = loadImages(path, folder1, targetList[i], width, height)
            target = i*np.ones(images.shape[0], dtype=np.int32)
            vars()[folder1+'_filenameList'].extend(filenames)
            vars()[folder1+'_input'].extend(images)
            vars()[folder1+'_target'].extend(target)
        vars()[folder1+'_input'] = np.asarray(vars()[folder1+'_input'])
        vars()[folder1+'_target'] = np.asarray(vars()[folder1+'_target'])
        
    print('\nPreprocessing time: ', time.time()-start_time, 'sec \n')
    
#%% parameter settings
    kernel_size = [5, 5, 5]     # CNN: kernel size (same length as num_filters)
    num_filters = [20, 30, 40]    # CNN: filters
    num_neurons = [200, 50]         # CNN: neurons in fully-connected layer
    num_epoch = 50
    batchSize = 100
    learningRate_max = 0.001
    learningRate_min = 0.0001
    learningRate_decay = 1000  # decay speed
    
#%% NN setup
    tf.InteractiveSession.close
    output_length = 10
    
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, width, height, 3], name='X')
    Y = tf.placeholder(tf.int32, [None, ], name='Y')
    drop_rate = tf.placeholder(tf.float32, name='drop_rate')
    lr = tf.placeholder(tf.float32, name='lr')
    
    with tf.variable_scope('ConvLayer', reuse=tf.AUTO_REUSE) as scope:
        namelist = list(np.arange(len(num_filters))+1)
        y = addConvLayer(X, num_filters[0], kernel_size[0], 'Conv'+str(namelist[0]))
        for a in range(len(num_filters)-1):
            y = addConvLayer(y, num_filters[a+1], kernel_size[a+1], 'Conv'+str(namelist[a+1]))
        y = tf.layers.Flatten()(y)
    with tf.variable_scope('FcLayer', reuse=tf.AUTO_REUSE) as scope:
        namelist = list(np.arange(len(num_neurons))+1)
        y = addFcLayer(y, y.shape[-1], num_neurons[0], 'Wi'+str(namelist[0]), 'B'+str(namelist[0]), drop_rate)
        for a in range(len(num_neurons)-1):
            y = addFcLayer(y, num_neurons[a], num_neurons[a+1], 
                          'W'+str(namelist[a])+str(namelist[a+1]), 'B'+str(namelist[a+1]), drop_rate)
        yo = addFcLayer(y, num_neurons[-1], output_length, 'W'+str(namelist[-1])+'o', 'Bo', drop_rate, activation_function = None)
        yo = tf.identity(yo, name = 'yo')
    loss = tf.losses.sparse_softmax_cross_entropy(Y, yo)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, name = 'optimizer')
    training_op = optimizer.minimize(loss)
    
    y_pred = tf.nn.softmax(yo)
    y_pred = tf.argmax(y_pred, axis=1, output_type=tf.int32, name='y_pred')
    y_true = Y
    correct = tf.cast(tf.equal(y_pred, y_true), tf.float32)
    accuracy = tf.reduce_mean(correct)
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
#%% train and test
    start_time = time.time()
    
    Epoch = []
    Loss_train = []
    Loss_test = []
    Accuracy_train = []
    Accuracy_test = []
    
    for epoch in range(num_epoch+1):
        input_train_shuffle, target_train_shuffle = randomShuffle(train_input, train_target)
        
        Epoch.append(epoch)
        Loss_train.append(sess.run(loss, feed_dict={X: train_input, Y: train_target, drop_rate: 0}))
        Loss_test.append(sess.run(loss, feed_dict={X: val_input, Y: val_target, drop_rate: 0}))
        Accuracy_train.append(sess.run(accuracy, feed_dict={X: train_input, Y: train_target, drop_rate: 0}))
        Accuracy_test.append(sess.run(accuracy, feed_dict={X: val_input, Y: val_target, drop_rate: 0}))
        print(epoch, "Training Loss:", Loss_train[epoch], "Testing Loss:", Loss_test[epoch])
            
        for iteration in range(len(target_train_shuffle) // batchSize):
            i = epoch * len(target_train_shuffle) // batchSize + iteration
            learning_rate = learningRate_min + (learningRate_max - learningRate_min) * math.exp(-i/learningRate_decay)
            X_batch = input_train_shuffle[iteration*batchSize:iteration*batchSize+batchSize,:]
            Y_batch = target_train_shuffle[iteration*batchSize:iteration*batchSize+batchSize]
            sess.run(training_op, feed_dict={X: X_batch, Y: Y_batch, drop_rate: 0.1, lr: learning_rate})
    
    Pred_test, True_test = sess.run([y_pred, y_true], feed_dict={X: val_input, Y: val_target, drop_rate: 0})
    print('\nTraining time: ', time.time()-start_time, 'sec \n')
    
#%% Plot
    # Fig1. Learning curve (Loss)
    plt.figure(figsize=(12,8))
    plt.plot(Epoch, Loss_train,'r',label="Train")
    plt.plot(Epoch, Loss_test,'g',label="Test")
    plt.yscale('log')
    plotType('Epoch', 'Loss', 'Learning curve (loss)')
    
    # Fig2. Learning curve (Accuracy)
    plt.figure(figsize=(12,8))
    plt.plot(Epoch, Accuracy_train,'r',label="Train")
    plt.plot(Epoch, Accuracy_test,'g',label="Test")
    plotType('Epoch', 'Accuracy', 'Learning curve (accuracy)')
    
    # Fig3. Show some examples and list the accuracy for each test classes
    correctly, incorrectly = selectExample(Pred_test, True_test, val_filenameList, targetList)
    
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(val_input[correctly[0]])#.reshape((width,height)))
    plt.xlabel(correctly[1])
    title = 'Label: '+correctly[2] + ',  Prediction: '+correctly[3]
    plt.title(title)
    
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(val_input[incorrectly[0]])#.reshape((width,height)))
    plt.xlabel(incorrectly[1])
    title = 'Label: '+incorrectly[2] + ',  Prediction: '+incorrectly[3]
    plt.title(title)
    
    class_Acc = listAccuracy(Pred_test, True_test)
    print('\nAccuracy of classes: \n')
    for i in range(output_length):
        print('{0:10}'.format(targetList[i]), ' : ', class_Acc[i], ' %')