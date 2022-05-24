from KETIToolDL.PredictionTool.inference import Inference
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class RNNStyleModelInfernce(Inference):
    # For small data without answer
    def __init__(self):
        self.batch_size = 1
    
    def setModel(self, trainParameter, model_method, modelFilePath):
        from KETIToolDL.TrainTool.trainer import RNNStyleModelTrainer as RModel
        IM= RModel()
        IM.setTrainParameter(trainParameter)
        IM.getModel(model_method)
        import torch
        self.infModel = IM.model
        self.infModel.load_state_dict(torch.load(modelFilePath[0]))

    def setData(self, data):
        self.data = data
    
    def get_result(self):
        yhat = self.infModel(self.data)
        result = yhat.to(device).detach().numpy()
        return result



class RNNStyleModelTestInference(Inference):
    # For TestData with answer
    def __init__(self):
        self.batch_size = 1
    
    def setModel(self, trainParameter, model_method, modelFilePath):
        from KETIToolDL.TrainTool.trainer import RNNStyleModelTrainer as RModel
        IM= RModel()
        IM.setTrainParameter(trainParameter)
        IM.getModel(model_method)
        import torch
        self.infModel = IM.model
        self.infModel.load_state_dict(torch.load(modelFilePath[0]))

    def setTestData(self, test, transformParameter):
        from torch.utils.data import DataLoader
        
        from KETIPreDataTransformation.trans_for_purpose.machineLearning import  LSTMData
        self.input_dim = len(transformParameter['feature_col'])
        LSTMD = LSTMData()
        testX_arr, testy_arr = LSTMD.transformXyArr(test, transformParameter)
        self.test_DataSet, self.test_loader = LSTMD.getTorchLoader(testX_arr, testy_arr, self.batch_size)
        #test_loader_one = DataLoader(test_DataSet, batch_size=1, shuffle=False, drop_last=True)


    def get_result(self):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in self.test_loader:
                x_test = x_test.view([self.batch_size, -1, self.input_dim]).to(device)
                y_test = y_test.to(device)
                self.infModel.eval()
                yhat = self.infModel(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())
        return predictions, values

