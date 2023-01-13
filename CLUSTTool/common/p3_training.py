import os, sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from KETIToolDL.CLUSTTool.common import p1_integratedDataSaving as p1
import pandas as pd
import matplotlib.pyplot as plt

def deleteLowQualityTrainValidationData(train, val, cleanTrainDataParam, integration_freq_sec, NaNProcessingParam):
    if cleanTrainDataParam =='Clean':
        import datetime
        # TODO integration_freq sec  사용을 안하는데 추후 문제될 수 있으니 확인해봐야 함
        #timedelta_frequency_sec = datetime.timedelta(seconds= integration_freq_sec)
        # 3. quality check
        from Clust.clust.quality.NaN import cleanData
        CMS = cleanData.CleanData()
        train = CMS.get_cleanData_by_removing_column(train, NaNProcessingParam) 
        val = CMS.get_cleanData_by_removing_column(val, NaNProcessingParam) 

    else:
        pass
    return train, val

def getTrainValData(data, featureList, scalerRootPath, splitRatio, scalerParam, scaleMethod ='minmax', mode = None, windows=None):
    trainval, scalerFilePath = getScaledData(scalerParam, scalerRootPath, data[featureList], scaleMethod)
    from Clust.clust.transformation.purpose import machineLearning as ML
    train, val = ML.splitDataByRatio(trainval, splitRatio, mode, windows)
    
    return train, val, scalerFilePath

def getScaledData(scalerParam, scalerRootpath, data, scaleMethod):
    if scalerParam=='scale':
        from Clust.clust.transformation.general.dataScaler import DataScaler
        DS = DataScaler(scaleMethod, scalerRootpath )
        #from Clust.clust.transformation.general import dataScaler
        #feature_col_list = dataScaler.get_scalable_columns(train_o)
        DS.setScaleColumns(list(data.columns))
        DS.setNewScaler(data)
        resultData = DS.transform(data)
        scalerFilePath = DS.scalerFilePath
    else:
        resultData = data.copy()
        scalerFilePath=None

    return resultData, scalerFilePath

def cleanNaNDF(dataSet, NaNProcessingParam, timedelta_frequency_sec):
    feature_cycle=NaNProcessingParam['feature_cycle']
    feature_cycle_times=NaNProcessingParam['feature_cycle_times']
    NanInfoForCleanData=NaNProcessingParam['NanInfoForCleanData']

    feature_list = dataSet.columns
    from Clust.clust.transformation.splitDataByCycle import dataByCycle

    dayCycle = dataByCycle.getCycleSelectDataSet(dataSet, feature_cycle, feature_cycle_times, timedelta_frequency_sec)


    from Clust.clust.quality.NaN import cleanData
    CMS = cleanData.CleanFeatureData(feature_list, timedelta_frequency_sec)
    refinedData, filterImputedData = CMS.getMultipleCleanDataSetsByDF(dayCycle, NanInfoForCleanData)
    CleanData = pd.concat(filterImputedData.values())

    return CleanData

def getModelFilePath(trainDataPathList, model_method):

    from KETIToolDL import modelInfo
    MI = modelInfo.ModelFileManager()
    modelFilePath = MI.getModelFilePath(trainDataPathList, model_method)
    return modelFilePath
    
def updateModelMetaData(ModelName, modelInfoMeta, trainModelMetaFilePath):
    modelMeta = p1.readJsonData(trainModelMetaFilePath)
    modelMeta[ModelName]=modelInfoMeta
    p1.writeJsonData(trainModelMetaFilePath, modelMeta)
    return modelInfoMeta
