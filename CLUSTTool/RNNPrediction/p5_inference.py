import setting
import pandas as pd
import p1_integratedDataSaving as p1
import p2_dataSelection as p2
import p3_training as p3
import p4_testing as p4

def inference(input, trainParameter, model_method, modelFilePath, scalerParam, scalerFilePath, featureList, target_col):
    scaleMethod='minmax'
    inputDF = pd.DataFrame(input, columns = featureList)
    # 4.Inference Data Preparation
    scaler = p4.getScalerFromFile(scalerFilePath)
    inputData= p4.getScaledData(inputDF, scaler, scalerParam)

    # 5. Inference
    from KETIToolDL.PredictionTool.RNNStyleModel.inference import RNNStyleModelInfernce
    Inference = RNNStyleModelInfernce()
    input_DTensor = Inference.getTensorInput(inputData)
    Inference.setData(input_DTensor)

    Inference.setModel(trainParameter, model_method, modelFilePath)
    inference_result = Inference.get_result()
    print(inference_result)
    if scalerParam =='scale':
        baseDFforInverse = pd.DataFrame(columns=featureList, index=range(1))
        baseDFforInverse[target_col] = inference_result[0]
        prediction_inverse = pd.DataFrame(scaler.transform(baseDFforInverse), columns=featureList, index=baseDFforInverse.index)
        result = prediction_inverse[target_col].values[0]
    else:
        result = inference_result[0][0]

    return result


if __name__ == "__main__":
    # (0) data selection
    DataMeta = p1.readJsonData(setting.DataMetaPath)
    dataList =  list(DataMeta.keys())
    dataName = dataList[2]
    dataSaveMode = DataMeta[dataName]["integrationInfo"]["DataSaveMode"]
    data = p2.getSavedIntegratedData(dataSaveMode, dataName)

    # 1. model selection
    ModelMeta =p1.readJsonData(setting.trainModelMetaPath)
    modelList = list(ModelMeta.keys())
    modelName = modelList[0]

    # 2. read ModelMeta
    past_step = ModelMeta[modelName]['transformParameter']['past_step']
    featureList = ModelMeta[modelName]['featureList']
    target_col = ModelMeta[modelName]['transformParameter']['target_col']
    scalerParam = ModelMeta[modelName]['scalerParam']
    scalerFilePath = ModelMeta[modelName]['scalerFilePath']
    modelFilePath = ModelMeta[modelName]['modelFilePath']
    trainParameter = ModelMeta[modelName]['trainParameter']
    model_method = ModelMeta[modelName]['model_method']
    

    # (0). Test Value
    data = data[featureList]
    inputData = data[-past_step:][featureList]

    result = inference(inputData, trainParameter, model_method, modelFilePath, scalerParam, scalerFilePath, featureList, target_col)
    print(result)

    
