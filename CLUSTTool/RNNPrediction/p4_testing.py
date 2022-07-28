
import sys
sys.path.append("../")

from KETIToolDL.CLUSTTool.common import p2_dataSelection as p2
from KETIToolDL.CLUSTTool.RNNPrediction import p3_training as p3

import pandas as pd 
def getScalerFromFile(scalerFilePath):
    import joblib
    scaler = joblib.load(scalerFilePath)
    return scaler

def getScaledData(data, scaler, scalerParam):

    if scalerParam=='scale':
        scaledD = pd.DataFrame(scaler.transform(data), index = data.index, columns = data.columns)
    else:
        scaledD = data.copy()
    return scaledD

def getClenaedData(data, cleanTrainDataParam, integration_freq_sec, NaNProcessingParam):
    if cleanTrainDataParam =='Clean':
        import datetime
        timedelta_frequency_sec = datetime.timedelta(seconds= integration_freq_sec)
        result = p3.cleanNaNDF(data, NaNProcessingParam,  timedelta_frequency_sec)

    else:
        pass
    
    return result
def getTestValues(test, trainParameter, transformParameter, model_method, modelFilePath):
    from KETIToolDL.PredictionTool.RNNStyleModel.inference import RNNStyleModelTestInference
    TestInference = RNNStyleModelTestInference()
    TestInference.setTestData(test, transformParameter)
    TestInference.setModel(trainParameter, model_method, modelFilePath)
    predictions, values = TestInference.get_result()

    return predictions, values


def getPredictionDFResult(predictions, values, scalerParam, scaler, featureList, target_col):
    if scalerParam =='scale':
        
        baseDFforInverse = pd.DataFrame(columns=featureList, index=range(len(predictions)))

        baseDFforInverse[target_col] = predictions
        prediction_inverse = pd.DataFrame(scaler.transform(baseDFforInverse), columns=featureList, index=baseDFforInverse.index)

        baseDFforInverse[target_col] = values 
        values_inverse = pd.DataFrame(scaler.transform(baseDFforInverse), columns=featureList, index=baseDFforInverse.index)

        df_result = pd.DataFrame(data={"value": values_inverse[target_col], "prediction": prediction_inverse[target_col]}, index=baseDFforInverse.index)
    else:
        df_result = pd.DataFrame(data={"value": values, "prediction": predictions}, index=range(len(predictions)))
    return df_result

def getTestResult(dataName, modelName, DataMeta, ModelMeta, dataRoot, db_client):

    dataSaveMode = DataMeta[dataName]["integrationInfo"]["DataSaveMode"]
    data = p2.getSavedIntegratedData(dataSaveMode, dataName, dataRoot, db_client)
    scalerFilePath = ModelMeta[modelName]["scalerFilePath"]
    featureList = ModelMeta[modelName]["featureList"]
    cleanTrainDataParam = ModelMeta[modelName]["cleanTrainDataParam"]
    scalerParam = ModelMeta[modelName]["scalerParam"]
    integration_freq_sec = ModelMeta[modelName]['trainDataInfo']["integration_freq_sec"]
    NaNProcessingParam = ModelMeta[modelName]['NaNProcessingParam']
    trainParameter = ModelMeta[modelName]["trainParameter"]
    transformParameter = ModelMeta[modelName]["transformParameter"]
    model_method = ModelMeta[modelName]["model_method"]
    modelFilePath = ModelMeta[modelName]["modelFilePath"]
    target_col = ModelMeta[modelName]["transformParameter"]["target_col"]

    scaler = getScalerFromFile(scalerFilePath)
    test = data[featureList]
    test = getScaledData(test, scaler, scalerParam)
    test = getClenaedData(test, cleanTrainDataParam, integration_freq_sec, NaNProcessingParam)
    
    prediction, values = getTestValues(test, trainParameter, transformParameter, model_method, modelFilePath)
    df_result = getPredictionDFResult(prediction, values, scalerParam, scaler, featureList, target_col)

    from KETIToolDataExploration.stats_table import metrics
    result_metrics =  metrics.calculate_metrics_df(df_result)

    return df_result, result_metrics