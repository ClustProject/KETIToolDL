
import torch
import os
from KETIToolDL.TrainTool.Brits import Brits_model
import copy 
import numpy as np
from sklearn.preprocessing import StandardScaler

from KETIToolDL.BatchTool.influxDBBatchTrainer import InfluxDBBatch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class BritsInference():
    def __init__(self, data, column_name, model_path):
        self.inputData = data
        self.column_name = column_name
        self.model_path = model_path
        #BritsModelFolder = os.path.join("/Users", "jw_macmini", "CLUSTGit", "DL", "Models",'brits','air_indoor_요양원', "ICL1L2000011",'in_ciai')

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

    def get_result(self):
        output = self.inputData.copy()
        if os.path.isfile(self.model_path[0]):
            print("Brits Model exists")
            print(self.model_path[0])
            loaded_model = Brits_model.Brits_i(108, 1, 0, len(output), device).to(device)
            loaded_model.load_state_dict(copy.deepcopy(torch.load(self.model_path[1], device)))
            
            Brits_model.makedata(output, self.model_path[0])
            data_iter = Brits_model.get_loader(self.model_path[0], batch_size=64)
            
            result = self.predict_result(loaded_model, data_iter, device, output)
            result_list = result.tolist()
            nan_data = output[output.columns[0]].isnull()
            for i in range(len(nan_data)):
                if nan_data.iloc[i] == True:
                    output[output.columns[0]].iloc[i] = result_list[i]
        else:
            print("No Brits Model File")
            pass
        
        return output
    
    def predict_result(self, model, data_iter, device, data):
        imputation = self.evaluate(model, data_iter, device)
        scaler = StandardScaler()
        scaler = scaler.fit(data[self.column_name].to_numpy().reshape(-1,1))
        result = scaler.inverse_transform(imputation[0])
        return result[:, 0]

    def evaluate(self, model, data_iter, device):
        model.eval()
        imputations = []
        for idx, data in enumerate(data_iter):
            data = Brits_model.to_var(data, device)
            ret = model.run_on_batch(data, None)
            eval_masks = ret['eval_masks'].data.cpu().numpy()
            imputation = ret['imputations'].data.cpu().numpy()
            imputations += imputation[np.where(eval_masks == 1)].tolist()
        imputations = np.asarray(imputations)
        return imputation