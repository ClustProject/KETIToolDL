import os
class Trainer():
    def __init__(self):
        pass

    def setBasicModelInfo(self, model_folder, model_name):
        self.model_folder = model_folder
        self.model_name = model_name
        
    def trainData(self, data):
        self.checkFolder(self.model_folder)
        self.setModelFilesName(self.model_name)
        self.trainSaveModel(data)
        print("Model Saved")

    def checkFolder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder) 

    ## Abstract 
    def setModelFilesName(self, model_name):
        pass

    def trainSaveModel(self, data):
        pass

    def setTrainParameter(self, parameter=None):
        self.parameter = parameter

# Model 1: Brits
from KETIToolDL.Model.Brits.training import BritsTraining
import torch
class BritsTrainer(Trainer):
    def setModelFilesName(self, model_name):
        self.model_name = model_name + '.pth'
        self.json_name = model_name  + '.json'
        self.model_path =os.path.join(self.model_folder, self.model_name)
        self.json_path = os.path.join(self.model_folder, self.json_name)

    def trainSaveModel(self, df): 
        Brits = BritsTraining(df, self.json_path)
        model = Brits.train()
        torch.save(model.state_dict(), self.model_path)
        print(self.model_path)

    def setTrainParameter(self, parameter=None):
        self.parameter = parameter
