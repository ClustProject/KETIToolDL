import os
class BatchTrainer:
    def __init__(self, parameter):
        self.parameter = parameter
        self.modelFileExtension = parameter['modelFileExtension']
        self.modelFullDirRootPath = self._getModelFullDirRootPath(parameter)

    def _getModelFullDirRootPath(self, parameter):

        modelFullDirRootPath = parameter['modelRootPath']
        for addPath in parameter['modelAddPathLink']:
            modelFullDirRootPath.append(parameter[addPath])

        print(modelFullDirRootPath)
        return modelFullDirRootPath

    def getModelFolderNameList(self, model_rootDir, model_name_list):
        model_folder = model_rootDir
        for model_name in model_name_list:
            model_folder.append(model_name)
        return model_folder

    def setTrainMethod(self, trainMethod):
        self.trainer = trainMethod