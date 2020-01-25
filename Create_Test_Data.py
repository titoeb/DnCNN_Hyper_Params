#!/usr/bin/env python
# coding: utf-8

import numpy as np

# Load data
X_test = np.load('/scratch2/ttoebro/data/X_test_rad41.npy')
Y_test = np.load('/scratch2/ttoebro/data/Y_test_rad41.npy')

# Create random samples for test data. Both undersampled (X) and fully sampled (Y)
np.random.seed(1993)
select = np.random.choice(np.arange(0, X_test.shape[0]), size = 300)

# Take sample from data
X_test = X_test[select, :, :, :]
Y_test = Y_test[select, :, :, :]

# Save the samples to file.
np.save('/scratch2/ttoebro/data/X_test_rad41_subset.npy', X_test)
np.save('/scratch2/ttoebro/data/Y_test_rad41_subset.npy', Y_test)

# RM matrices from memory.
del X_test, Y_test

# Load training data
X_train = np.load('/scratch2/ttoebro/data/X_train_rad41.npy')

# Create a random sample from the undersampled images
np.random.seed(1993)
select = np.random.choice(np.arange(0, X_train.shape[0]), size = 15000)

X_train = X_train[select, :, :, :]

# Save the sample to file.
np.save('/scratch2/ttoebro/data/X_train_rad41_subset.npy', X_train)

# RM matrix from main memory.
del X_train

# Create sample from fully sample train images.
Y_train = np.load('/scratch2/ttoebro/data/Y_train_rad41.npy')

Y_train = Y_train[select, :, :, :]

np.save('/scratch2/ttoebro/data/Y_train_rad41_subset.npy', Y_train)

