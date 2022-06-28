from KETIToolDL.PredictionTool.inference import Inference
import torch
import torch.nn as nn
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")

class ClassificationModelInference(Inference):
    def __init__(self):
        super().__init__()
        
    def setModel(self):
        pass
    
    def setTestData(self, test, transformParameter):
        pass
    
    def test(model, test_loader):
        """
        Predict classes for test dataset based on the trained model

        :param model: best trained model
        :type model: model

        :param test_loader: test dataloader
        :type test_loader: DataLoader

        :return: predicted classes
        :rtype: numpy array

        :return: prediction probabilities
        :rtype: numpy array

        :return: test accuracy
        :rtype: float
        """

        model.eval()   # 모델을 validation mode로 설정
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            corrects = 0
            total = 0
            preds = []
            probs = []
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.long)

                # forwinputs = inputs.to(device)ard
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

                #preds.extend(pred.to(device).detach().numpy()) 
                #probs.extend(prob.to(device).detach().numpy())

            preds = np.array(preds)
            probs = np.array(probs)
            acc = (corrects.double() / total).item()
        
        return preds, probs, acc

    def get_loadersForTest(x_test, y_test, min_class, batch_size):
        """
        :param x_test: x test data
        :type x_test: 
        
        :param y_test: y_test data
        :type y_test: 
        
        """
        y_test = y_test - min_class

        # test 데이터셋 구축
        datasets = []
        for dataset in [(x_test, y_test)]:
            x_data = np.array(dataset[0])
            y_data = dataset[1]
            datasets.append(torch.utils.data.TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data)))

        testset =  datasets[0]
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        return test_loader
