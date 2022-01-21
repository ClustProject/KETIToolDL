import os
import sys
sys.path.append("..")
sys.path.append("../..")

class Trainer():
    def __init__(self):
        pass
    
    def setTrainParameter(self, parameter=None):
        self.parameter = parameter

    def trainModel(self, model_folder, model_name, input):
        self.inputData  = input 
        self._checkModelFolder(model_folder)
        self.trainData = self._processInputData(self.inputData)
        self._setModelFilesName(model_folder, model_name)
        self._trainSaveModel(self.trainData)
        print("Model Saved")

    def _checkModelFolder(self, model_folder):
        if not os.path.exists(model_folder):
            os.makedirs(model_folder) 
        
    def _processInputData(self, inputData):
        trainData = inputData.copy()
        trainData = self._preprocessData(trainData)
        trainData = self._transformData(trainData)
        return trainData

    ## TODO
    def _preprocessData(self, data):
        result = data
        return result

    def _transformData(self, data):
        result = data
        return result

    ## Abstract 
    def _setModelFilesName(self, model_folder, model_name):
        pass

    def _trainSaveModel(self, data):
        pass

# Model 1: Brits
from KETIToolDL.Model.Brits.training import BritsTraining
import torch
class BritsTrainer(Trainer):
    def _setModelFilesName(self, model_folder, model_name):
        self.model_name = model_name + '.pth'
        self.json_name = model_name  + '.json'
        self.model_path =os.path.join(model_folder, self.model_name)
        self.json_path = os.path.join(model_folder, self.json_name)

    def _trainSaveModel(self, df): 
        Brits = BritsTraining(df, self.json_path)
        model = Brits.train()
        torch.save(model.state_dict(), self.model_path)
        print(self.model_path)

