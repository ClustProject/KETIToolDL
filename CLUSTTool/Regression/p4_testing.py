import sys
sys.path.append("../")

from KETIToolDL.CLUSTTool.common import p4_testing as p4
def getResultMetrics(test_x, test_y, model_method, target, modelFilePath, scalerParam, scaler_y, trainParameter, batch_size, device):
    from KETIToolDL.TrainTool.Regression.trainer import RegressionML as RML

    rml = RML(model_method, trainParameter)
    model = rml.getModel()

    from KETIToolDL.PredictionTool.Regression.inference import RegressionModelTestInference as RTI
    ri = RTI(test_x, test_y, batch_size, device)
    print(test_y)
    ri.transInputDFtoNP()
    print(ri.X[0], ri.y[0])
    pred, mse, mae = ri.get_result(model, modelFilePath)
    print(pred)
    df_result = p4.getPredictionDFResult(pred, test_y.values, scalerParam, scaler_y, featureList= target, target_col = target[0])
    print(df_result)
    from KETIToolDataExploration.stats_table import metrics
    result_metrics =  metrics.calculate_metrics_df(df_result)
    return df_result, result_metrics
