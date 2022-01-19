import os
import sys
import torch
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("../../../..")

from KETIToolDL.trainBySource import influxDBData 
from KETIToolDL.Model.Brits import training

class BritsInfluxTraining(influxDBData.trainModels):
    def setModelFileName(self, target_name):
        self.model_name = target_name + '.pth'
        self.json_name = target_name  + '.json'
        self.model_path =os.path.join(self.model_folder, self.model_name)
        self.json_path = os.path.join(self.model_folder, self.json_name)

    def trainSaveModel(self, df): 
        Brits = training.BritsTraining(df, self.json_path)
        model = Brits.train()
        torch.save(model.state_dict(), self.model_path)
        print(self.model_path)


