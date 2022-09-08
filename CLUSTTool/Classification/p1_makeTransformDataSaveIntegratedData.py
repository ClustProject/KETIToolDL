import os
import sys
import json
import pandas as pd
import numpy as np
import torch

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../..")

#from KETIToolDL.CLUSTTool.common import p1_integratedDataSaving as p1
from KETIToolDL.CLUSTTool.common import p1_integratedDataSaving as p1


def make_different_freq_data(data, splitNum, transform_freqlist, freqTransformMode): ## db ms name 어쩔겨?
    columns = data.columns
    split_dataset = {}
    for num in range(splitNum):
        data_c = data.copy()
        print(data.info())
        ## 서로 다른 주기 별 데이터 생성
        if freqTransformMode == "drop":
            ## data frequency transform
            split_data = data.iloc[[idx for idx in data.index if idx%transform_freqlist[num]==0]]
            ## set data index
            split_data.set_index(['datetime'], inplace = True)
        else: # freqTransformMode == "sampling"
            ## set data index
            data_c.set_index(['datetime'], inplace = True)
            split_data = data_c.resample(str(transform_freqlist[num])+'S').mean()

        ## get split data
        split_data = split_data[columns[splitNum*num:splitNum*(num+1)]]

        ## get split data set
        split_dataset[num] = split_data
        
        print("split num : ", num)
        print("split data shape : ", split_data.shape)
        print("------")
    return split_dataset


def getIntegratedDataFrom3array(original_dataset, db_client, splitNum, rename_columns, transform_freqlist, startTime, endTime, 
                                original_freq, dataInfo, processParam, transformParam, integration_freq_sec, integration_method, 
                                integration_duration_criteria, dataReadMode, freqTransformMode):
    
    count = 0
    int_dataset = pd.DataFrame()
    seq_len = original_dataset.shape[2]
    print("dataset shape : ",original_dataset.shape)
    
    for array_X in original_dataset:
        print("array num : ", count)
        print("......................")
        
        # 3array -> 2array -> dataframe
        data_x_trans = pd.DataFrame(array_X)
        data_x = data_x_trans.T
        data_x.rename(columns = rename_columns, inplace = True)
        timeIndex = pd.date_range(start=startTime, freq = original_freq, periods=seq_len)
        data_x['datetime'] = timeIndex
        
        # make split dataSet
        dataSet = make_different_freq_data(data_x, splitNum, transform_freqlist, freqTransformMode)
        count+=1

        # get integrated data
        int_data = p1.getData(db_client, dataInfo, integration_freq_sec, 
                          processParam, startTime, endTime, integration_method, transformParam, 
                          integration_duration_criteria, dataReadMode, dataSet)
        
        # get integrated data length
        windows = len(int_data)
        print("++++++++ integrated data length : ", windows, "++++++++")
        # concat
        int_dataset = pd.concat([int_dataset, int_data])
    
    # integrated data set index
    timeIndex2 = pd.date_range(start=startTime, freq = str(integration_freq_sec)+"S", periods=len(int_dataset))
    int_dataset.set_index(timeIndex2, inplace = True)
    
    return int_dataset, windows