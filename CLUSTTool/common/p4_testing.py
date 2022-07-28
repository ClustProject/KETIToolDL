
import os, sys
sys.path.append("../")

from KETIToolDL.CLUSTTool.common import p3_training as p3

import pandas as pd 
def getScalerFromFile(scalerFilePath):
    import joblib
    scaler = joblib.load(scalerFilePath)
    return scaler

def getScaledData(data, scaler, scalerParam):
    scaleMethod='minmax'
    if scalerParam=='scale':
        scaledD = pd.DataFrame(scaler.transform(data), index = data.index, columns = data.columns)
    else:
        scaledD = data.copy()
    return scaledD

def getPredictionDFResult(predictions, values, scalerParam, scaler, featureList, target_col):
    print(scalerParam)
    if scalerParam =='scale':
        
        baseDFforInverse = pd.DataFrame(columns=featureList, index=range(len(predictions)))

        baseDFforInverse[target_col] = predictions
        prediction_inverse = pd.DataFrame(scaler.inverse_transform(baseDFforInverse), columns=featureList, index=baseDFforInverse.index)
        baseDFforInverse[target_col] = values 
        values_inverse = pd.DataFrame(scaler.inverse_transform(baseDFforInverse), columns=featureList, index=baseDFforInverse.index)
        df_result = pd.DataFrame(data={"value": values_inverse[target_col], "prediction": prediction_inverse[target_col]}, index=baseDFforInverse.index)
    else:
        df_result = pd.DataFrame(data={"value": values, "prediction": predictions}, index=range(len(predictions)))
    return df_result

def getCleandData(data, cleanTrainDataParam, integration_freq_sec, NaNProcessingParam):
    if cleanTrainDataParam =='Clean':
        import datetime
        timedelta_frequency_sec = datetime.timedelta(seconds= integration_freq_sec)
        result = p3.cleanNaNDF(data, NaNProcessingParam,  timedelta_frequency_sec)

    else:
        result = data.copy()
        pass
    
    return result