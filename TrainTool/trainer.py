import os
import sys
sys.path.append("..")
sys.path.append("../..")

class Trainer():
    def __init__(self):
        from multiprocessing import freeze_support
        freeze_support() 
        pass
    
    def setTrainParameter(self, parameter=None):
        self.parameter = parameter

    def trainModel(self, input, modelFilePath):
        self.inputData  = input 
        self.trainData = self.processInputData(self.inputData)
        self.modelFilePath = modelFilePath 
        self._trainSaveModel(self.trainData)
        print("Model Saved")

    def processInputData(self, inputData):
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
        Brits = BritsTraining(df, self.modelFilePath[0])
        model = Brits.train()
        torch.save(model.state_dict(), self.modelFilePath[1])
        print(self.modelFilePath)

