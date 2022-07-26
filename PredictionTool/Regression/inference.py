
import torch
import numpy as np
from KETIToolDL.PredictionTool.inference import Inference
from sklearn.metrics import mean_absolute_error, mean_squared_error 
import torch.nn as nn

class RegressionModelInference(Inference):

    def __init__(self, device):
        self.batch_size = 1
        self.device = device
    
    def setModel(self, model, modelFilePath):
        import torch
        self.infModel = model
        self.infModel.load_state_dict(torch.load(modelFilePath[0]))
        self.infModel.eval()

    def getTensorInput(self, data):
        """The method get tensor input for training.

        dataframe -> ndarray size = (past_step, len(feature_col_list)) -> 
        ndarray size = (1, past_step, len(feature_col_list))  -> torch tensor

        Note:
            tensor input transformation

        Args:
            data (pd.DataFrame): input dataframe

        Returns:
            data (torch.utils.data.DataLoader): input tensor data
        """
        # TODO
        if type(data) == 'object': #dataFrame ??
            inference_input = data.values.astype(np.float32)
        else:
            inference_input = data
        #inference_input = inference_input.reshape((-1, inference_input.shape[0], inference_input.shape[1]))
        inference_input = data
        print(inference_input.shape)
        inference_input_tensor = torch.tensor(inference_input)
        return inference_input_tensor

    def setData(self, data):
        self.data = data
  
    def get_result(self):
        yhat = self.infModel(self.data)
        result = yhat.to(self.device).detach().numpy()
        return result


class RegressionModelTestInference(Inference):

    def __init__(self, X, y, batch_size, device):
        self.X = X
        self.y = y 
        self.batch_size = batch_size
        self.test_loader = self.get_testLoader()
        self.device = device

    def get_testLoader(self):
        
        x_data = np.array(self.X)
        y_data = self.y
        testData= torch.utils.data.TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data))
        test_loader = torch.utils.data.DataLoader(testData, batch_size=self.batch_size, shuffle=True)
        return test_loader

    def get_result(self, init_model, best_model_path):
        """
        Predict class based on the best trained model
        :param init_model: initialized model
        :type model: model

        :param best_model_path: path for loading the best trained model
        :type best_model_path: str

        :return: predicted values
        :rtype: numpy array

        :return: test mse
        :rtype: float

        :return: test mae
        :rtype: float
        """

        print("\nStart testing data\n")

        # load best model
        init_model.load_state_dict(torch.load(best_model_path[0]))

        # get prediction and accuracy
        pred, mse, mae = self.test(init_model, self.test_loader)
        print(f'** Performance of test dataset ==> MSE = {mse}, MAE = {mae}')
        print(f'** Dimension of result for test dataset = {pred.shape}')
        return pred, mse, mae

    def test(self, model, test_loader):
        """
        Predict classes for test dataset based on the trained model

        :param model: best trained model
        :type model: model

        :param test_loader: test dataloader
        :type test_loader: DataLoader

        :return: predicted values
        :rtype: numpy array

        :return: test mse
        :rtype: float

        :return: test mae
        :rtype: float
        """

        model.eval()   # 모델을 validation mode로 설정
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            trues, preds = [], []

            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device, dtype=torch.float)

                # forward
                # input을 model에 넣어 output을 도출
                outputs = model(inputs)
                
                # 예측 값 및 실제 값 축적
                trues.extend(labels.detach().cpu().numpy())
                preds.extend(outputs.detach().cpu().numpy())
        
        preds = np.array(preds).reshape(-1)
        trues = np.array(trues)

        mse = mean_squared_error(trues, preds)
        mae = mean_absolute_error(trues, preds)
        return preds, mse, mae

    