#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

# Here the Grid search is executed.
def run_grid(grid_loc,  #Location of the grid to run
             directory, #Location of the models that should be run 
                  name,  #Names of the models
                  steps, # How many steps should each model be run
                  batch_size # With what batch sizes should the model be runned
                    ):
    
    # Load grid to run
    data = pd.read_csv(grid_loc, dtype={'Error_message': str, 'location': str})
    
    # Load data for model
    # X_train = np.load('/scratch2/ttoebro/data/X_test_pois_1_9.npy')
    #Y_train = np.load('/scratch2/ttoebro/data/Y_test_pois_1_9.npy')
    #X_test = np.load('/scratch2/ttoebro/data/X_test_pois_1_9.npy')
    #Y_test = np.load('/scratch2/ttoebro/data/Y_test_pois_1_9.npy')
    
    # Run each model specification in the rows if the 'location' columns is empty
    for row in range(0, data.shape[0]):
        if(data['location'][row] != ''):
            # Run Model specification
            try:
                res = Execute_model(data.loc[row, 'depth'],
                      data.loc[row, 'filter_num'],
                      data.loc[row, 'loss_scheme'],
                      data.loc[row, 'skipped_scheme'],
                      data.loc[row, 'filter_size'],
                      data.loc[row, 'learning_rate'],
                      directory,
                      str(row),
                      name,
                      batch_size,
                              steps)
            except Exception as e:
                print(e)
                e = "Error in Execute model. " + str(e)
                data.loc[row, 'Avg_MAE'] = -12.0
                data.loc[row, 'Avg_MSE'] = -12.0
                data.loc[row, 'Avg_SSIM'] = -12.0
                data.loc[row, 'Runtime_test'] = -3.0
                data.loc[row, 'Runtime_train'] = -3.0
                data.loc[row, 'Error_message'] = e
                data.loc[row, 'location'] = '-.-'
                next()
                
            data.loc[row, 'Avg_MAE'] = res['Avg_MAE']
            data.loc[row, 'Avg_MSE'] = res['Avg_MSE']
            data.loc[row, 'Avg_SSIM'] = res['Avg_SSIM']
            data.loc[row, 'Runtime_test'] = res['Runtime_test']
            data.loc[row, 'Runtime_train'] = res['Runtime_train']
            data.loc[row, 'Error_message'] = res['Error_message']
            data.loc[row, 'location'] = directory + name + "_" + str(row)
            # Save results from run to data
          
            # Save progress
            data.to_csv(grid_loc)
            
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
    print("filter_size is " + str(filter_size) )

    # return MAE, MSE, SSIM on test and run-time
    return({'Avg_MAE': np.random.random(), 'Avg_MSE': np.random.random(), 'Avg_SSIM' : np.random.random(), 'Runtime_test' : np.random.random() * 10, 'Runtime_train' : np.random.random() * 20, 'Error_message' : "No Error occured."})   


# Load the grid search main file
data = pd.read_csv('/scratch2/ttoebro/Grid_search/Grid/main_grid.csv' , dtype={'Error_message': str, 'location': str})
data.to_csv('/scratch2/ttoebro/Grid_search/Grid/main_grid.csv', index=False)

# Execute the grid search.
run_grid('/scratch2/ttoebro/Grid_search/Grid/main_grid.csv',  '/scratch2/ttoebro/Grid_search/Grid',  'DnCNN',  10, 8 )

