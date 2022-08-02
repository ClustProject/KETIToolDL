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
    test_y, scaler_y = p4.getScaledTestData(datay[target], y_scalerFilePath, scalerParam)
    # 4. Testing
    batch_size=1
    df_result, result_metrics = getResultMetrics(test_x, test_y, model_method, target, modelFilePath, scalerParam, scaler_y, trainParameter, batch_size, device, windowNum)
    return df_result, result_metrics

def getResultMetrics(test_x, test_y, model_method, target, modelFilePath, scalerParam, scaler_y, trainParameter, batch_size, device, windowNum=0):
    from KETIToolDL.TrainTool.Regression.trainer import RegressionML as RML

    rml = RML(model_method, trainParameter)
    model = rml.getModel()

    from KETIToolDL.PredictionTool.Regression.inference import RegressionModelTestInference as RTI
    ri = RTI(test_x, test_y, batch_size, device)
    ri.transInputDFtoNP(windowNum)
    pred, trues, mse, mae = ri.get_result(model, modelFilePath)
    df_result = p4.getPredictionDFResult(pred, trues, scalerParam, scaler_y, featureList= target, target_col = target[0])
    df_result.index = test_y.index
    from KETIToolDataExploration.stats_table import metrics
    result_metrics =  metrics.calculate_metrics_df(df_result)
    return df_result, result_metrics
