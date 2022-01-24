import os
import sys
sys.path.append("..")
sys.path.append("../..")

class Trainer():
    def __init__(self):
        pass
    
    def setTrainParameter(self, parameter=None):
        self.parameter = parameter

    def trainModel(self, model_folder, model_name, input, modelFileExtension):
        self.inputData  = input 
        self._checkModelFolder(model_folder)
        self.trainData = self._processInputData(self.inputData)
        self._setModelFilesName(model_folder, model_name, modelFileExtension)
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

    def _setModelFilesName(self, model_folder, model_name, modelFileExtension):
        self.model_path=[]
        for i, fileExtension in enumerate(modelFileExtension):
            temp = model_name + fileExtension
            self.model_path.append(os.path.join(model_folder, temp))
        print(self.model_path)
        
    ## Abstract 
    def _trainSaveModel(self, data):
        pass

# Model 1: Brits
from KETIToolDL.Model.Brits.training import BritsTraining
import torch
class BritsTrainer(Trainer):
    def _trainSaveModel(self, df): 
        Brits = BritsTraining(df, self.model_path[0])
        model = Brits.train()
        torch.save(model.state_dict(), self.model_path[1])
        print(self.model_path)

