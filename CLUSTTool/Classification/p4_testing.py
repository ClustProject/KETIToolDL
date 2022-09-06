import sys
sys.path.append("../")

from KETIToolDL.CLUSTTool.common import p4_testing as p4
from KETIToolDL.CLUSTTool.common import p2_dataSelection as p2
import os
"""
def getTestResult(dataName_X, dataName_y, modelName, DataMeta, ModelMeta, dataFolderPath, currentFolderPath, device, windowNum=0, db_client=None):

    dataSaveMode_X = DataMeta[dataName_X]["integrationInfo"]["DataSaveMode"]
    dataSaveMode_y = DataMeta[dataName_y]["integrationInfo"]["DataSaveMode"]
    dataX = p2.getSavedIntegratedData(dataSaveMode_X, dataName_X, dataFolderPath)
    datay = p2.getSavedIntegratedData(dataSaveMode_y, dataName_y, dataFolderPath)
    X_scalerFilePath = os.path.join(currentFolderPath, ModelMeta[modelName]['files']['XScalerFile']["filePath"])
    y_scalerFilePath = os.path.join(currentFolderPath, ModelMeta[modelName]['files']['yScalerFile']["filePath"])
    modelFilePath_old = ModelMeta[modelName]['files']['modelFile']["filePath"]
    modelFilePath =[]
    for modelFilePath_one in modelFilePath_old:
        modelFilePath.append(os.path.join(currentFolderPath, modelFilePath_one))
    featureList = ModelMeta[modelName]["featureList"]
    target = ModelMeta[modelName]["target"]
    scalerParam = ModelMeta[modelName]["scalerParam"]
    model_method = ModelMeta[modelName]["model_method"]
    trainParameter = ModelMeta[modelName]["trainParameter"]
    

    # Scaling Test Input
    test_x, scaler_X = p4.getScaledTestData(dataX[featureList], X_scalerFilePath, scalerParam)
    test_y, scaler_y = p4.getScaledTestData(datay[target], y_scalerFilePath, scalerParam)
    # 4. Testing
    batch_size=1
    df_result, result_metrics = getResultMetrics(test_x, test_y, model_method, target, modelFilePath, scalerParam, scaler_y, trainParameter, batch_size, device, windowNum)
    return df_result, result_metrics

"""
def getTestResult(dataName_X, dataName_y, modelName, DataMeta, ModelMeta, dataFolderPath, device, windowNum=0, db_client=None):

    dataSaveMode_X = DataMeta[dataName_X]["integrationInfo"]["DataSaveMode"]
    dataSaveMode_y = DataMeta[dataName_y]["integrationInfo"]["DataSaveMode"]
    dataX = p2.getSavedIntegratedData(dataSaveMode_X, dataName_X, dataFolderPath)
    datay = p2.getSavedIntegratedData(dataSaveMode_y, dataName_y, dataFolderPath)
    
    if modelName.split("_")[1] == "ML":
        datay = datay.iloc[1:]
    
    X_scalerFilePath = ModelMeta[modelName]['files']['XScalerFile']["filePath"]
    y_scalerFilePath = ModelMeta[modelName]['files']['yScalerFile']["filePath"]
    modelFilePath = ModelMeta[modelName]['files']['modelFile']["filePath"]

    featureList = ModelMeta[modelName]["featureList"]
    target = ModelMeta[modelName]["target"]
    scalerParam = ModelMeta[modelName]["scalerParam"]
    model_method = ModelMeta[modelName]["model_method"]
    trainParameter = ModelMeta[modelName]["trainParameter"]
    # Scaling Test Input

    test_x, scaler_X = p4.getScaledTestData(dataX[featureList], X_scalerFilePath, scalerParam)
    test_y, scaler_y = p4.getScaledTestData(datay[target], y_scalerFilePath, scalerParam)# No Scale
    # test_y = datay[target] # for classification

    # 4. Testing
    batch_size=1
    scalerParam="noScale" # for classification
    df_result, result_metrics, acc = getResultMetrics(test_x, test_y, model_method, target, modelFilePath, scalerParam, scaler_y, trainParameter, batch_size, device, windowNum)
    return df_result, result_metrics, acc


def getResultMetrics(test_x, test_y, model_method, target, modelFilePath, scalerParam, scaler_y, trainParameter, batch_size, device, windowNum=0):
    import pandas as pd
    from KETIToolDL.TrainTool.Classification.trainer import ClassificationML as CML

    cml = CML(model_method, trainParameter)
    model = cml.getModel()

    from KETIToolDL.PredictionTool.Classification.inference import ClassificationModelTestInference as CTI
    ci = CTI(test_x, test_y, batch_size, device)
    ci.transInputDFtoNP(windowNum)
    pred, prob, trues, acc = ci.get_result(model, modelFilePath)
    from sklearn.metrics import classification_report
    result_metrics = classification_report(trues, pred, output_dict = True)
    
    df_result = p4.getPredictionDFResult(pred, trues, scalerParam, scaler_y, featureList= target, target_col = target[0])
    df_result.index = test_y.index
    
    return df_result, result_metrics, acc

 