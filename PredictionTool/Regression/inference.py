
import torch
import numpy as np
from KETIToolDL.PredictionTool.inference import Inference
import torch.nn as nn

class RegressionInference(Inference):

    def __init__(self, X, y, batch_size, parameter):
        self.test_loader = self.get_testLoader(X, y, batch_size)
        self.parameter = parameter

    def get_testLoader(self, X, y, batch_size):
        
        x_data = np.array(X)
        y_data = y
        testData= torch.utils.data.TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data))
        test_loader = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True)
        return test_loader


    def get_result(self, init_model, best_model_path):
        """
        Predict class based on the best trained model
        :param init_model: initialized model
        :type model: model

        :param best_model_path: path for loading the best trained model
        :type best_model_path: str

        :return: predicted classes
        :rtype: numpy array

        :return: prediction probabilities
        :rtype: numpy array

        :return: test accuracy
        :rtype: float
        """

        print("\nStart testing data\n")

        # load best model
        init_model.load_state_dict(torch.load(best_model_path))
        model = init_model
        test_loader = self.test_loader

        #### Test
        model.eval()   # 모델을 validation mode로 설정
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            corrects = 0
            total = 0
            preds = []
            probs = []
            for inputs, labels in test_loader:
                inputs = inputs.to(self.parameter['device'])
                labels = labels.to(self.parameter['device'], dtype=torch.long)

                # forward
                # input을 model에 넣어 output을 도출
                outputs = model(inputs)
                prob = outputs
                prob = nn.Softmax(dim=1)(prob)
                
                # output 중 최댓값의 위치에 해당하는 class로 예측을 수행
                _, pred = torch.max(outputs, 1)
                
                # batch별 정답 개수를 축적함
                corrects += torch.sum(pred == labels.data)
                total += labels.size(0)

                preds.extend(pred.detach().cpu().numpy())
                probs.extend(prob.detach().cpu().numpy())

            preds = np.array(preds)
            probs = np.array(probs)
            acc = (corrects.double() / total).item()
       
        return preds, probs, acc

    