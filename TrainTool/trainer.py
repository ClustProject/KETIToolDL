import os
import sys
sys.path.append("..")
sys.path.append("../..")

class Trainer():
    def __init__(self):
        from multiprocessing import freeze_support
        freeze_support() 
        pass
    
    def getModel(self, model_method):
        self.model_method = model_method

    def setTrainParameter(self, modelParameter=None):
        """
        modelParameter is dictonary Type, its format is dependent on atrain method/
        
        example
        >>> modelParameter = {'input_dim': input_dim,
                'hidden_dim' : hidden_dim,
                'layer_dim' : layer_dim,
                'output_dim' : output_dim,
                'dropout_prob' : dropout}

        """
        self.modelParameter = modelParameter

    # Very Important Main Function
    def trainModel(self, input, modelFilePath):
        """
        1. get input and model file path
        2. train data and make model
        3. save model

        """
        self.inputData  = input 
        self.trainData = self.processInputData(self.inputData)
        self.modelFilePath = modelFilePath 
        self._trainSaveModel(self.trainData)
        print("Model Saved")

    def processInputData(self, inputData):
        trainData = inputData.copy()
        return trainData
     
    ## Abstract 
    def _trainSaveModel(self, data):
        pass

# Model 1: Brits
from KETIToolDL.TrainTool.Brits.training import BritsTraining
import torch
import torch.nn as nn
import torch.optim as optim

# Model 1: Brits
class BritsTrainer(Trainer):
    def _trainSaveModel(self, df): 
        Brits = BritsTraining(df, self.modelFilePath[0])
        model = Brits.train()
        torch.save(model.state_dict(), self.modelFilePath[1])
        print(self.modelFilePath)


# Model 2: RNN 계열
class RNNStyleModelTrainer(Trainer):
    def processInputData(self, train, val, transformParameter, cleanParam, batch_size):
        self.batch_size = batch_size
        from KETIPreDataTransformation.trans_for_purpose.machineLearning import  LSTMData
        LSTMD = LSTMData()
        trainX_arr, trainy_arr = LSTMD.transformXyArr(train, transformParameter, cleanParam)
        self.train_DataSet, self.train_loader = LSTMD.getTorchLoader(trainX_arr, trainy_arr,  batch_size)

        valX_arr, valy_arr = LSTMD.transformXyArr(val, transformParameter, cleanParam)
        self.val_DataSet, self.val_loader = LSTMD.getTorchLoader(valX_arr, valy_arr, batch_size)

    def getModel(self, model_method):
        super().getModel(model_method)
        from KETIToolDL.TrainTool.RNN import model
        models = {
            "rnn": model.RNNModel,
            "lstm": model.LSTMModel,
            "gru": model.GRUModel,
        }
        self.model = models.get(model_method.lower())(**self.modelParameter)
        return self.model

    def saveModel(self): 
        """
        torch.save(model, model_rootPath + 'model.pt')  # 전체 모델 저장
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, model_rootPath + 'all.tar')  # 여러 가지 값 저장, 학습 중 진행 상황 저장을 위해 epoch, loss 값 등 일반 scalar값 저장 가능
        """
        print(self.modelFilePath[0])
        torch.save(self.model.state_dict(), self.modelFilePath[0])  # 모델 객체의 state_dict 저장
        print(self.modelFilePath)

    def trainModel(self, n_epochs, modelFilePath):
        from KETIToolDL.TrainTool.RNN.optimizer import Optimization
        #from torch import optim
        self.modelFilePath = modelFilePath
        # Optimization
        weight_decay = 1e-6
        learning_rate = 1e-3
        loss_fn = nn.MSELoss(reduction="mean")

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        opt = Optimization(model=self.model, loss_fn=loss_fn, optimizer=self.optimizer)

        opt.train(self.train_loader, self.val_loader, batch_size=self.batch_size, n_epochs=n_epochs, n_features=self.modelParameter['input_dim'])
        opt.plot_losses()
        self.opt = opt
        # 모델의 state_dict 출력
        #self.printState_dict()
        self.saveModel()
    
    def printState_dict(self):
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

        # 옵티마이저의 state_dict 출력
        print("Optimizer's state_dict:")
        for var_name in self.optimizer.state_dict():
            print(var_name, "\t", self.optimizer.state_dict()[var_name])


