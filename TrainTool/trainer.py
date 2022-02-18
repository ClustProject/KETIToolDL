import os
import sys
sys.path.append("..")
sys.path.append("../..")

class Trainer():
    def __init__(self):
        pass
    
    def setTrainParameter(self, parameter=None):
        self.parameter = parameter

    def trainModel(self, input, PathInfo):
        self.inputData  = input 
        self.modelFolderpath = self.getModelFolder(PathInfo)
        self.trainData = self._processInputData(self.inputData)
        self._setModelFilesName(self.modelFolderpath, PathInfo['ModelFileName'])
        self._trainSaveModel(self.trainData)
        print("Model Saved")

    def getModelFolder(self, PathInfo):
        modelFolderpath =''
        for add_folder in PathInfo['ModelRootPath']:
            modelFolderpath = os.path.join(modelFolderpath, add_folder)
        for add_folder in PathInfo['ModelInfoPath']:
            modelFolderpath = os.path.join(modelFolderpath, add_folder)
        for add_folder in PathInfo['TrainDataPath']:
            modelFolderpath = os.path.join(modelFolderpath, add_folder)
        self._checkModelFolder(modelFolderpath)

        return modelFolderpath

    def _checkModelFolder(self, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path) 
    
    def _setModelFilesName(self, model_path, model_name_list):
        self.model_path=[]
        for i, model_name in enumerate(model_name_list):
            self.model_path.append(os.path.join(model_path, model_name))
        print(self.model_path)

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
    def _trainSaveModel(self, data):
        pass

# Model 1: Brits
from KETIToolDL.TrainTool.Brits.training import BritsTraining
import torch
class BritsTrainer(Trainer):
    def _trainSaveModel(self, df): 
        Brits = BritsTraining(df, self.model_path[0])
        model = Brits.train()
        torch.save(model.state_dict(), self.model_path[1])
        print(self.model_path)

