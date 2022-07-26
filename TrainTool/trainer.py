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


import numpy as np
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

class RegressionInferenceML():
    def __init__(self, model_name):
        pass

import numpy as np
class RegressionML(Trainer):
    
    def __init__(self, model_name, parameter):
        """
        Set initial parameter and model name for training

        :param model_name: Model Name
        :type model_name: String
        
        :param parameter: parameter
        :type parameter: Dictionary
        """
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

        self.device = parameter['device']
        self.model_name = model_name
        self.parameter = parameter


    def processInputData(self, train_X, train_y, batch_size):
        """
        Prepare and set Input Data
        :param trainX: train_X data
        :type trainX: Array
        
        :param trainy: train_y data
        :type trainy: Array
        
        :param batch_size: batch_size
        :type batch_size: integer
    
        """
        
        # load dataloder
        
        X= train_X
        y = train_y
        
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
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
        
    
    def getModel(self):
        """
        Build model and return initialized model for selected model_name
    
        """
        from KETIToolDL.TrainTool.Regression.lstm_fcn import LSTM_FCNs
        from KETIToolDL.TrainTool.Regression.rnn import RNN_model
        from KETIToolDL.TrainTool.Regression.cnn_1d import CNN_1D
        from KETIToolDL.TrainTool.Regression.fc import FC

        """
        Build model and return initialized model for selected model_name

        :return: initialized model
        :rtype: model
        """

        # build initialized model
        if self.model_name == 'LSTM_rg':
            init_model = RNN_model(
                rnn_type='lstm',
                input_size=self.parameter['input_size'],
                hidden_size=self.parameter['hidden_size'],
                num_layers=self.parameter['num_layers'],
                bidirectional=self.parameter['bidirectional'],
                device=self.parameter['device']
            )
        elif self.model_name == 'GRU_rg':
            init_model = RNN_model(
                rnn_type='gru',
                input_size=self.parameter['input_size'],
                hidden_size=self.parameter['hidden_size'],
                num_layers=self.parameter['num_layers'],
                bidirectional=self.parameter['bidirectional'],
                device=self.parameter['device']
            )
        elif self.model_name == 'CNN_1D_rg':
            init_model = CNN_1D(
                input_channels=self.parameter['input_size'],
                input_seq=self.parameter['seq_len'],
                output_channels=self.parameter['output_channels'],
                kernel_size=self.parameter['kernel_size'],
                stride=self.parameter['stride'],
                padding=self.parameter['padding'],
                drop_out=self.parameter['drop_out']
            )
        elif self.model_name == 'LSTM_FCNs_rg':
            init_model = LSTM_FCNs(
                input_size=self.parameter['input_size'],
                num_layers=self.parameter['num_layers'],
                lstm_drop_p=self.parameter['lstm_drop_out'],
                fc_drop_p=self.parameter['fc_drop_out']
            )
        elif self.model_name == 'FC_rg':
            init_model = FC(
                representation_size=self.parameter['input_size'],
                drop_out=self.parameter['drop_out'],
                bias=self.parameter['bias']
            )
        else:
            print('Choose the model correctly')
        return init_model

    def trainModel(self, init_model, modelFilePath, num_epochs):
        """
        Train model and return best model

        :param init_model: initialized model
        :type init_model: model

        :param modelFilePath: model file path to be saved
        :type modelFilePath: string

        :param num_epochs: number of epochs
        :type modelFilePath: integer

        :return: best trained model
        :rtype: model
        """

        print("Start training model")


        # train model
        init_model = init_model.to(self.device)

        dataloaders_dict = {'train': self.train_loader, 'val': self.valid_loader}
        criterion = nn.MSELoss()
        optimizer = optim.Adam(init_model.parameters(), lr=self.parameter['lr'])

        self.best_model = self.train(init_model, dataloaders_dict, criterion, num_epochs, optimizer, self.parameter['device'])
        self._trainSaveModel(self.best_model, modelFilePath)
        return self.best_model
    

    def train(self, model, dataloaders, criterion, num_epochs, optimizer, device):
        import time
        import copy
        """
        Train the model

        :param model: initialized model
        :type model: model

        :param dataloaders: train & validation dataloaders
        :type dataloaders: dictionary

        :param criterion: loss function for training
        :type criterion: criterion

        :param num_epochs: the number of train epochs
        :type num_epochs: int

        :param optimizer: optimizer used in training
        :type optimizer: optimizer

        :return: trained model
        :rtype: model
        """

        since = time.time()

        val_mse_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_mse = 10000000

        for epoch in range(num_epochs):
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print()
                print('Epoch {}/{}'.format(epoch + 1, num_epochs))

            # 각 epoch마다 순서대로 training과 validation을 진행
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # 모델을 training mode로 설정
                else:
                    model.eval()   # 모델을 validation mode로 설정

                running_loss = 0.0
                running_total = 0

                # training과 validation 단계에 맞는 dataloader에 대하여 학습/검증 진행
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device, dtype=torch.float)
                    
                    # parameter gradients를 0으로 설정
                    optimizer.zero_grad()

                    # forward
                    # training 단계에서만 gradient 업데이트 수행
                    with torch.set_grad_enabled(phase == 'train'):
                        # input을 model에 넣어 output을 도출한 후, loss를 계산함
                        outputs = model(inputs)
                        outputs = outputs.squeeze(1)
                        loss = criterion(outputs, labels)

                        # backward (optimize): training 단계에서만 수행
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # batch별 loss를 축적함
                    running_loss += loss.item() * inputs.size(0)
                    running_total += labels.size(0)

                # epoch의 loss 및 accuracy 도출
                epoch_loss = running_loss / running_total

                if epoch == 0 or (epoch + 1) % 10 == 0:
                    print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                # validation 단계에서 validation loss가 감소할 때마다 best model 가중치를 업데이트함
                if phase == 'val' and epoch_loss < best_mse:
                    best_mse = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_mse_history.append(epoch_loss)

        # 전체 학습 시간 계산
        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val MSE: {:4f}'.format(best_mse))

        # validation loss가 가장 낮았을 때의 best model 가중치를 불러와 best model을 구축함
        model.load_state_dict(best_model_wts)
        return model     

    def _trainSaveModel(self, best_model, best_model_path):
        """
        Save the best trained model

        :param best_model: best trained model
        :type best_model: model

        :param best_model_path: path for saving model
        :type best_model_path: str
        """

        # save model
        torch.save(best_model.state_dict(), best_model_path[0])

    
    