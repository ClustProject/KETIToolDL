from KETIToolDL.TrainTool.trainer import Trainer
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
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


    def processInputData(self, train_x, train_y, val_x, val_y, batch_size, windowNum = 0):
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
        
        from Clust.clust.transformation.type.DFToNPArray import transDFtoNP
        train_x, train_y = transDFtoNP(train_x, train_y, windowNum)
        val_x, val_y = transDFtoNP(val_x, val_y, windowNum)

        self.parameter['input_size'] = train_x.shape[1]
        self.parameter['seq_len']  = train_x.shape[2] # seq_length



        ## TODO 아래 코드 군더더기 저럴 필요 없음 어짜피 이 함수는 Train을 넣으면 Train, Valid 나누는 함수로 고정시키 때문에
        # train/validation 데이터셋 구축
        datasets = []
        for dataset in [(train_x, train_y), (val_x, val_y)]:
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

    
    