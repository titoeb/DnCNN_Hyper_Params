#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import gc
import time
import datetime
from PIL import Image
from SSIM_PIL import compare_ssim
import sys

def conv_layer(tensor_in, name_layer, is_training, f_num, f_size):
    x = tf.layers.conv2d(
        inputs = tensor_in,
        filters = f_num,
        kernel_size = [f_size, f_size],
        padding = "same",
        activation= None,
        name = name_layer,
        use_bias=False)
    
    x = tf.layers.batch_normalization(x, name = name_layer + "_bn",
                                             center=True, 
                                             scale=True, 
                                             training=is_training)
    
    return tf.nn.relu(x, name = name_layer + "_relu")
def conv_layer_without_relu(tensor_in, name_layer, is_training, f_num, f_size):
    x = tf.layers.conv2d(
        inputs = tensor_in,
        filters = f_num,
        kernel_size = [f_size, f_size],
        padding = "same",
        activation= None,
        name = name_layer,
        use_bias=False)
    
    x = tf.layers.batch_normalization(x, name = name_layer + "_bn",
                                             center=True, 
                                             scale=True, 
                                             training=is_training)
    
    return x

def DnCNN_model(features, labels, mode):
   # Input Layer
    input_oiginal = features['x']
   
    # Convolutional layer #1     
    input_layer = tf.layers.conv2d(
        inputs = input_oiginal,
        filters = filter_num_in,
        kernel_size = filter_size_in,
        padding = "same",
        activation= tf.nn.relu,
        name = "Conv_Init")
    is_training_mode = (mode == tf.estimator.ModeKeys.TRAIN)
    
    if (skipped_scheme_in == "ResNet"):
        cur_tensor = input_layer
        for i in range(1, depth_in + 1):
            if i == 1:
                cur_tensor = conv_layer(cur_tensor, "conv{}".format(i), is_training_mode, filter_num_in, filter_size_in)
                save = cur_tensor
            elif (i - 1) % 2 == 0:
                cur_tensor = conv_layer_without_relu(cur_tensor, "conv{}".format(i), is_training_mode, filter_num_in, filter_size_in)
                cur_tensor = tf.add(cur_tensor, save, name  = "conv{}".format(i) + "_add")
                cur_tensor = tf.nn.relu(cur_tensor, name = "conv{}".format(i) + "_relu")
                save = cur_tensor
            else:
                cur_tensor = conv_layer(cur_tensor, "conv{}".format(i), is_training_mode, filter_num_in, filter_size_in)
                
        # Final layers: Make it a gray scale image again!
        final_layer = tf.layers.conv2d(
        inputs = cur_tensor + save,
        filters =  1,
        kernel_size = [1, 1],
        padding = "same",
        activation = None,
        name = "final_layer")

    else:
        # Standard DnCNN skipped layers scheme: one residual link between first and last
        cur_tensor = input_layer
        for i in range(1, depth_in + 1):
            cur_tensor = conv_layer(cur_tensor, "conv{}".format(i), is_training_mode, filter_num_in, filter_size_in)

        # Final layers: Make it a gray scale image again!
        final_layer = tf.layers.conv2d(
            inputs = cur_tensor + input_layer,
            filters = 1,
            kernel_size = [1, 1],
            padding = "same",
            activation = None,
            name = "final_layer") 
      
    # Give output in prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions=final_layer)

    # Calulate loss function according to loss_scheme
    if(loss_scheme_in == 'SSIM'):
        loss = -tf.reduce_mean(tf.image.ssim(labels, final_layer, 1.0))
    elif(loss_scheme_in == 'L1'):
        loss = tf.losses.mean_squared_error(labels = labels , predictions = final_layer)
    else:
        loss = tf.losses.absolute_difference(labels = labels , predictions = final_layer)
       
    ### Print summary ###
    if (mode != tf.estimator.ModeKeys.PREDICT):
        tf.summary.image("Undersampled", input_oiginal, max_outputs = 1)
        tf.summary.image("Output_Image", final_layer, max_outputs = 1)
        tf.summary.image("True_Image", labels,  max_outputs = 1)
        tf.summary.histogram("Summary_final_layer", final_layer)
        tf.summary.histogram("Summary_labels", labels)
        tf.summary.scalar("Value_Loss_Function", loss)
        
        # Besides the specific ones write out all trainiable variables
        for var in tf.trainable_variables():
            name = var.name
            name = name.replace(':', '_')
            tf.summary.histogram(name, var)
            
        # Finally merge the summary
        merged_summary = tf.summary.merge_all()

    # Specify Learning 
    if mode == tf.estimator.ModeKeys.TRAIN:
        # This is needed so that Batch normalization paramters are trained as well.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Specify the Optimizer
            original_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate_in)
            # Use gradient clipping to avoid exploding gradients
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=5.0)
            # The loss function should be minimized.
            train_op = optimizer.minimize(loss = loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
def run_model(directory, name, model_id, batch_size, steps):
    runconf = tf.estimator.RunConfig(save_summary_steps=5, log_step_count_steps = 10, tf_random_seed= 1993)
    DnCNN = tf.estimator.Estimator(config=runconf,
        model_fn=DnCNN_model, model_dir = directory + name + "_" + model_id)

    train = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_train},
        y=Y_train,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=False)
    
    DnCNN.train(input_fn=train, steps=steps)

# Define the evaluation of the model.
def eval_model(directory, name, model_id, batch_size, steps):
    # Specify Model
    runconf = tf.estimator.RunConfig(save_summary_steps=1000, log_step_count_steps = 1000, tf_random_seed= 1993)
    DnCNN = tf.estimator.Estimator(config=runconf,
        model_fn=DnCNN_model, model_dir = directory + name + "_" + model_id)
    
    # Evaluate 
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_test[:,:,:,:]},
        y=Y_test[:,:,:,:],
        batch_size = 1,
        shuffle=False)
    predict_results = DnCNN.predict(input_fn=predict_input_fn)
    
    sum_mae = 0
    sum_mse = 0
    sum_ssim = 0
    
    #for im_num in range(0, Y_test.shape[0]):
    for im_num in range(0, 10):
        prediction = next(predict_results)
        true_image = Y_test[im_num,:,:,:]
        sum_mae += np.mean(np.abs(prediction - true_image))
        sum_mse += np.mean(np.power((prediction - true_image), 2))
        sum_ssim += compare_ssim(Image.fromarray((prediction[:,:,0] * 255).astype('uint8'), 'L'),
                 Image.fromarray((true_image[:,:,0] * 255).astype('uint8'), 'L'))
    
    mean_mae = sum_mae/ X_train.shape[0]
    mean_mse = sum_mse / X_train.shape[0]
    mean_ssim = sum_ssim / X_train.shape[0]
    return([mean_mae, mean_mse, mean_ssim])

def Execute_model(depth,
                  filter_num,
                  loss_scheme,
                  skipped_scheme,
                  filter_size,
                  learning_rate,
                  directory,
                  model_id,
                  name,
                  batch_size,
                  steps):
    # Check arguments
    if (depth < 1 or depth > 40):
        raise Exception('Depth of {} is not allowed!'.format(depth))
    if(filter_num < 1 or filter_num > 256):
        raise Exception('filter_num of {} is not allowed!'.format(filter_num))
    if(not (loss_scheme == "L1" or loss_scheme == "L2" or loss_scheme == "SSIM")):
        raise Exception('loss_scheme {} is not allowed!'.format(loss_scheme))
    if(not (skipped_scheme == 'DnCNN' or skipped_scheme == 'ResNet')):
        raise Exception('skipped_scheme {} is not allowed!'.format(skipped_scheme))
    if(filter_size < 1 or filter_size > 9):
        raise Exception('filter_size of {} is not allowed!'.format(filter_size))
    if(learning_rate < 0 or learning_rate > 1):
        raise Exception('learning_rate of {} is not allowed!'.format(learning_rate))
        sum_ssim
        
    # Define arguments global
    global depth_in, filter_num_in, loss_scheme_in, skipped_scheme_in, filter_size_in, learning_rate_in, directory_in, model_id_in, name_in, batch_size_in, steps_in
    depth_in = depth
    filter_num_in = filter_num
    loss_scheme_in = loss_scheme
    skipped_scheme_in = skipped_scheme
    filter_size_in = filter_size
    learning_rate_in = learning_rate
    directory_in = directory
    model_id_in = model_id
    name_in = name
    batch_size_in = batch_size
    steps_in = steps
        
    ## run model ##
    Runtime_test = -1.0
    Runtime_train = -1.0
    start = time.time()
    error_message = ' '
    try:
        run_model(directory, name, model_id, batch_size, steps)
        Runtime_train = time.time() - start
        start = time.time()
        try:
            eval_results = eval_model(directory, name, model_id, batch_size, steps)
            MAE = eval_results[0]
            MSE = eval_results[1]
            SSIM = eval_results[2]
        except Exception as e:
            print("Error in testing. " + str(e))
            error_message =  "Error in testing. " + str(e)
            MAE = -10.0
            MSE =  -10.0
            SSIM =  -10.0
        Runtime_test = time.time() - start
    except Exception as e:
        print("Error in trainnig. " + str(e))
        error_message = "Error in trainnig. " + str(e)
        MAE = -10.0
        MSE = -10.0
        SSIM = -10.0
        
    # Remove global variables
    del depth_in, filter_num_in, loss_scheme_in, skipped_scheme_in, filter_size_in, learning_rate_in, directory_in, model_id_in, name_in, batch_size_in, steps_in
    if error_message == ' ':
        error_message = "No Error occured."
    # return MAE, MSE, SSIM on test and run-time
    return({'Avg_MAE': MAE, 'Avg_MSE': MSE, 'Avg_SSIM' : SSIM, 'Runtime_test' : Runtime_test, 'Runtime_train' : Runtime_train, "Error_message" : error_message})   

# Load the training and test data.
X_train = np.load("/scratch2/ttoebro/data/X_test_pois_1_9.npy")
X_test = np.load("/scratch2/ttoebro/data/X_test_pois_1_9.npy")
Y_train = np.load("/scratch2/ttoebro/data/Y_test_pois_1_9.npy")
Y_test = np.load("/scratch2/ttoebro/data/Y_test_pois_1_9.npy")

# Test the model.
Execute_model(10, 16, 'L1', 'ResNet',  5,  0.1,  '/scratch2/ttoebro/models/test/', 'Resnet', 'this6', 3, 2000)
