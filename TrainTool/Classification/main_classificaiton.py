import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from models.train_model import Train_Test
from models.lstm_fcn import LSTM_FCNs
from models.rnn import RNN_model
from models.cnn_1d import CNN_1D
from models.fc import FC

import warnings
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")

class Classification():
    def __init__(self, trainX, trainy,  model_name, parameter, model_path, trainParameter) :
        """
        Initialize Classification class and prepare dataloaders for training and testing

        :param train_data: train data with X and y
        :type train_data: dictionary

        :param model_name: model training method
        :type model_name: string

        :param parameter: parameter
        :type parameter: dictionary

        :model_path: model save path
        :model_path: string

        
        example
            >>> 
        """

        self.model = model_name
        self.parameter = parameter

        self.input_size = trainX.shape[1] # input size
        if len(trainX.shape) == 3:
            self.seq_length = trainX.shape[2] # seq_length
        else:
            self.seq_length = 0
    
        self.min_class = self.getClassRealMinNumber(trainy)

        self.model_path = model_path
        self.trainParameter = trainParameter

        # load dataloder
        batch_size = self.trainParameter['batch_size'] 
        self.train_loader, self.valid_loader = self.get_loaders(trainX, trainy, batch_size=batch_size)

        

        # build trainer
        self.trainer = Train_Test(self.parameter, self.train_loader, self.valid_loader)

    def build_model(self, num_classes):
        """
        Build model and return initialized model for selected model_name

        :param num_classes: number of classes
        :type model_name: integer

        :return: initialized model
        :rtype: model
        """
        # build initialized model
        if self.model == 'LSTM':
            init_model = RNN_model(
                rnn_type='lstm',
                input_size=self.input_size,
                num_classes=num_classes,
                hidden_size=self.parameter['hidden_size'],
                num_layers=self.parameter['num_layers'],
                bidirectional=self.parameter['bidirectional'],
                device=device
            )
        elif self.model == 'GRU':
            init_model = RNN_model(
                rnn_type='gru',
                input_size=self.input_size,
                num_classes=num_classes,
                hidden_size=self.parameter['hidden_size'],
                num_layers=self.parameter['num_layers'],
                bidirectional=self.parameter['bidirectional'],
                device=device
            )
        elif self.model == 'CNN_1D':
            init_model = CNN_1D(
                input_channels=self.input_size,
                num_classes=num_classes,
                input_seq=self.parameter['seq_len'],
                output_channels=self.parameter['output_channels'],
                kernel_size=self.parameter['kernel_size'],
                stride=self.parameter['stride'],
                padding=self.parameter['padding'],
                drop_out=self.parameter['drop_out']
            )
        elif self.model == 'LSTM_FCNs':
            init_model = LSTM_FCNs(
                input_size= self.input_size,
                num_classes=num_classes,
                num_layers=self.parameter['num_layers'],
                lstm_drop_p=self.parameter['lstm_drop_out'],
                fc_drop_p=self.parameter['fc_drop_out']
            )
        elif self.model == 'FC':
            init_model = FC(
                representation_size=self.input_size,
                num_classes=num_classes,
                drop_out=self.parameter['drop_out'],
                bias=self.parameter['bias']
            )
        else:
            print('Choose the model correctly')
        return init_model

    def train_model(self, init_model):
        """
        Train model and return best model

        :param init_model: initialized model
        :type init_model: model

        :return: best trained model
        :rtype: model
        """

        print("Start training model")

        # train model
        init_model = init_model.to(device)

        dataloaders_dict = {'train': self.train_loader, 'val': self.valid_loader}
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(init_model.parameters(), lr=self.parameter['lr'])

        best_model = self.trainer.train(init_model, dataloaders_dict, criterion, self.trainParameter, optimizer)
        return best_model

    def save_model(self, best_model, best_model_path):
        """
        Save the best trained model

        :param best_model: best trained model
        :type best_model: model

        :param best_model_path: path for saving model
        :type best_model_path: str
        """

        # save model
        torch.save(best_model.state_dict(), best_model_path)

    
    def getClassRealMinNumber(self, datay):
        # class의 값이 0부터 시작하지 않으면 0부터 시작하도록 변환
        min_class = 0
        if np.min(datay) != 0:
            min_class = np.min(datay)
            print('Set start class as zero')
            
        return min_class 

    def get_loaders(self, X, y, batch_size):
        """
        Get train, validation, and test DataLoaders
        
        :param train_data: train data with X and y
        :type train_data: dictionary

    
        :param batch_size: batch size
        :type batch_size: int

        :return: train, validation, and test dataloaders
        :rtype: DataLoader
        """
        
        # class의 값이 0부터 시작하지 않으면 0부터 시작하도록 변환
        y = y - self.min_class


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



def pred_data(init_model, best_model_path):
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
        pass
        