#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

# Function to create grid searches as csv for various hyper parameters.
def make_grid(direc,
                name,
                max_len,
                depth,
                filter_num,
                loss_scheme,
                skipped_scheme,
                filter_size,
                learning_rate):
    data_dict = {'model_id' : list(), 'depth' : list(), 'filter_num' : list(), 'loss_scheme' : list(), 'skipped_scheme' : list(), 'filter_size' : list(), 'learning_rate' : list()}
    iterator = 0
    for dep in depth:
        for filter_n in filter_num:
            for loss_s in loss_scheme:
                for skipped_sch in skipped_scheme:
                    for filter_s in filter_size:
                        for learning_r in learning_rate:
                            data_dict['depth'].append(dep)
                            data_dict['filter_num'].append(filter_n)
                            data_dict['loss_scheme'].append(loss_s)
                            data_dict['skipped_scheme'].append(skipped_sch)
                            data_dict['filter_size'].append(filter_s)
                            data_dict['learning_rate'].append(learning_r)
                            data_dict['model_id'].append(str(iterator))
                            iterator += 1
    data = pd.DataFrame(data_dict)
    data['Avg_MAE'] = -11.0
    data['Avg_MSE'] = -11.0
    data['Avg_SSIM'] = -11.0
    data['Runtime_test'] = -2.0
    data['Runtime_train'] = -2.0
    data['Error_message'] = "not_run"
    data['location'] = "not_run"
    if data.shape[0] <= max_len:
        data.to_csv(direc + name + ".csv", index=False)
    else:
        # separate df to different parts
        count = 0
        first = 0
        end = max_len
        while(end < data.shape[0]):
            tmp = data.iloc[first:end,:]
            tmp.to_csv(direc + name + "_part_" + str(count) + ".csv", index=False)
            count += 1
            first+= max_len
            end+= max_len

# Create the grid and store it.
make_grid(direc = '/scratch2/ttoebro/Grid_search/Grid/',
                name = 'main_grid',
                max_len = 48,
                depth = [10, 13, 16, 18, 20, 22, 25],
                filter_num = [16, 32, 64, 128],
                loss_scheme = ['L1', 'L2', 'SSIM'],
                skipped_scheme = ['DnCNN', 'ResNet'],
                filter_size = [3, 5, 7],
                learning_rate = [0.075])



