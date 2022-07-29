import sys
sys.path.append("../")

from KETIToolDL.CLUSTTool.common import p4_testing as p4
def getResultMetrics(test_x, test_y, model_method, target, modelFilePath, scalerParam, scaler_y, trainParameter, batch_size, device):
    from KETIToolDL.TrainTool.Regression.trainer import RegressionML as RML

    rml = RML(model_method, trainParameter)
    model = rml.getModel()

    from KETIToolDL.PredictionTool.Regression.inference import RegressionModelTestInference as RTI
    ri = RTI(test_x, test_y, batch_size, device)
    ri.transInputDFtoNP()
    pred, trues, mse, mae = ri.get_result(model, modelFilePath)
    df_result = p4.getPredictionDFResult(pred, trues, scalerParam, scaler_y, featureList= target, target_col = target[0])
    from KETIToolDataExploration.stats_table import metrics
    result_metrics =  metrics.calculate_metrics_df(df_result)
    return df_result, result_metrics
