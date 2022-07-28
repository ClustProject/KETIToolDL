import pandas as pd
import os, sys
sys.path.append("../")

# 
from KETIToolDL.CLUSTTool.common import p1_integratedDataSaving as p1



def trainSaveModel(trainModelMetaPath, train, val,  dataName, cleanTrainDataParam, NaNProcessingParam, transformParameter, scalerParam,  model_method,  trainParameter, batch_size, n_epochs, trainDataInfo, modelTags, trainDataType, modelPurpose, scalerFilePath):

    from KETIPreDataTransformation.general_transformation.dataScaler import encodeHashStyle
    transformParameter_encode =  encodeHashStyle(str(transformParameter))
    trainDataPathList = ["CLUST", dataName, transformParameter_encode]

    from KETIToolDL.TrainTool.trainer import RNNStyleModelTrainer as RModel
    RM= RModel()
    RM.processInputData(train, val, transformParameter, cleanTrainDataParam, batch_size)

    from KETIToolDL import modelInfo
    MI = modelInfo.ModelFileManager()
    modelFilePath = MI.getModelFilePath(trainDataPathList, model_method)

    RM.setTrainParameter(trainParameter)
    RM.getModel(model_method)
    RM.trainModel(n_epochs, modelFilePath)

    modelMeta = p1.readJsonData(trainModelMetaPath)
    modelName, modelInfoMeta = setModelData(dataName, modelMeta, trainDataInfo, modelTags, cleanTrainDataParam, NaNProcessingParam, transformParameter, scalerParam, trainParameter, transformParameter_encode, model_method, trainDataType, modelPurpose, scalerFilePath, modelFilePath)
    p1.writeJsonData(trainModelMetaPath, modelMeta)

    return modelName, modelFilePath, modelInfoMeta

def setModelData(dataName, modelMeta, trainDataInfo, modelTags, cleanTrainDataParam, NaNProcessingParam, transformParameter, scalerParam, trainParameter, transformParameter_encode, model_method, trainDataType, modelPurpose, scalerFilePath, modelFilePath):
    
    from KETIPreDataTransformation.general_transformation.dataScaler import encodeHashStyle
    ModelName = encodeHashStyle (p1.getListMerge ([transformParameter_encode, dataName] ))

    modelInfoMeta ={}
    modelInfoMeta ["trainDataInfo"] = trainDataInfo
    modelInfoMeta ["featureList"] = transformParameter["feature_col"]
    modelInfoMeta ["trainDataType"] = trainDataType
    modelInfoMeta ["modelPurpose"] = modelPurpose
    modelInfoMeta ["model_method"] = model_method
    modelInfoMeta ["modelTags"] = modelTags
    modelInfoMeta ["cleanTrainDataParam"] = cleanTrainDataParam
    modelInfoMeta ["NaNProcessingParam"] = NaNProcessingParam
    modelInfoMeta ["dataName"]=dataName
    modelInfoMeta ["transformParameter"]=transformParameter
    modelInfoMeta ["scalerParam"] =scalerParam
    modelInfoMeta ["scalerFilePath"] = scalerFilePath
    modelInfoMeta ["modelFilePath"] =modelFilePath
    modelInfoMeta ["trainParameter"] =trainParameter

    modelMeta[ModelName]=modelInfoMeta
    return ModelName, modelInfoMeta
