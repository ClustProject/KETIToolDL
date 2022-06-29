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

        opt.train(self.train_loader, self.val_loader, batch_size=self.batch_size, n_epochs=n_epochs, n_features=self.trainParameter['input_dim'])
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


import numpy as np
import pandas as pd
class ClassificationML(Trainer):
    def __init__(self):
        import random
        # seed 고정
        random_seed = 42

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)
        super().__init__()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"{self.device}" " is available.")

    def processInputData(self, trainX, trainy, model_name, trainParameter):
        """
        :param trainX: train x data
        :type trainX: Array
        
        :param trainy: train y data
        :type trainy: Array
        
        :param model_name: model training method
        :type model_name: string
        
        :param modelParameter: 선택한 Model의 Parameter
        :type modelParamter: dictionary
        
        :param trainParameter: 모델 학습 Parameter
        :type trainParameter: dictionary
        """
        from KETIToolDL.TrainTool.Classification.models.train_model import Train_Test
        
        self.model = model_name
        self.trainParameter = trainParameter
        
        self.input_size = trainX.shape[1] # input size
        if len(trainX.shape) == 3:
            self.seq_length = trainX.shape[2] # seq_length
        else:
            self.seq_length = 0

        # load dataloder
        batch_size = self.trainParameter['batch_size'] 
        self.train_loader, self.valid_loader = self.get_loaders(trainX, trainy, batch_size=batch_size)

        # build trainer
        self.trainer = Train_Test(self.train_loader, self.valid_loader)

    def get_loaders(self, X, y, batch_size):
        """
        Get train, validation, and test DataLoaders
        
        - y class의 label 값이 0부터 시작한다는 가정하에 분류를 진행
        
        :param train_data: train data with X and y
        :type train_data: dictionary

        :param batch_size: batch size
        :type batch_size: int

        :return: train, validation, and test dataloaders
        :rtype: DataLoader
        """

        # train data를 시간순으로 8:2의 비율로 train/validation set으로 분할
        n_train = int(0.8 * len(X))
        x_train, y_train = X[:n_train], y[:n_train]
        x_valid, y_valid = X[n_train:], y[n_train:]

        ## TODO 아래 코드 군더더기 저럴 필요 없음 어짜피 이 함수는 Train을 넣으면 Train, Valid 나누는 함수로 고정시키 때문에
        # train/validation 데이터셋 구축
        datasets = []
        for dataset in [(x_train, y_train), (x_valid, y_valid)]:
            x_data = np.array(dataset[0])
            y_data = dataset[1]
            datasets.append(torch.utils.data.TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data)))

        # train/validation DataLoader 구축
        trainset, validset = datasets[0], datasets[1]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
        
        return train_loader, valid_loader

    def setTrainParameter(self, num_classes, modelParameter=None):
        """
        Set model parameter
        
        :param num_classes: number of classes
        :type model_name: integer
        
        :param modelParameter: model parameter, its format is dependent on a train method
        :type modelParameter: dictionary
                
        example
        >>> modelParameter = {
                                'num_layers': 2,
                                'hidden_size': 64,
                                'dropout': 0.1,
                                'bidirectional': True,
                                'lr': 0.0001
                            }

        """
        
        self.modelParameter = modelParameter
        self.modelParameter["input_size"] = self.input_size 
        self.modelParameter["num_classes"] = num_classes # 처음 입력 arg에 num_classes 를 넣어도 됨
        
        if self.model == 'LSTM_cf':
            self.modelParameter["rnn_type"] = 'lstm'
            self.modelParameter["device"] = self.device
        elif self.model == 'GRU_cf':
            self.modelParameter["rnn_type"] = 'gru'
            self.modelParameter["device"] = self.device
       
    def getModel(self):
        """
        Build model and return initialized model for selected model_name
        """
        from KETIToolDL.TrainTool.Classification.models.lstm_fcn import LSTM_FCNs
        from KETIToolDL.TrainTool.Classification.models.rnn import RNN_model
        from KETIToolDL.TrainTool.Classification.models.cnn_1d import CNN_1D
        from KETIToolDL.TrainTool.Classification.models.fc import FC
        
        # build initialized model
        if (self.model == 'LSTM_cf') | (self.model == "GRU_cf"):
            self.init_model = RNN_model(**self.modelParameter)
        elif self.model == 'CNN_1D_cf':
            self.init_model = CNN_1D(**self.modelParameter)
        elif self.model == 'LSTM_FCNs_cf':
            self.init_model = LSTM_FCNs(**self.modelParameter)
        elif self.model == 'FC_cf':
            self.init_model = FC(**self.modelParameter)
        else:
            print('Choose the model correctly')
            
        return self.init_model
    
    def trainModel(self, modelFilePath):
        """
        Train model and return best model

        :return: best trained model
        :rtype: model
        """

        print("Start training model")
        
        self.modelFilePath = modelFilePath
        # train model
        self.init_model = self.init_model.to(self.device)

        dataloaders_dict = {'train': self.train_loader, 'val': self.valid_loader}
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.init_model.parameters(), lr=self.modelParameter['lr'])
        num_epochs = self.trainParameter["num_epochs"]

        best_model = self.trainer.train(self.init_model, dataloaders_dict, criterion, num_epochs, optimizer, self.device)
        
        self.saveModel(best_model)
        
        return best_model
        
    def saveModel(self, best_model):
        """
        Save the best trained model

        :param best_model: best trained model
        :type best_model: model

        """

        # save model
        torch.save(best_model.state_dict(), self.modelFilePath[0])

