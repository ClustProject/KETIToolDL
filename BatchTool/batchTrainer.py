import os
class BatchTrainer:
    def getModelFolderName(self, model_rootDir, model_name_list):
        model_folder = model_rootDir
        for model_name in model_name_list:
            model_folder = os.path.join(model_folder, model_name) 
        return model_folder