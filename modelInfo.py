import sys
sys.path.append("../")
sys.path.append("../..")
import os
from KETIToolDL import setting
# This class includes a general model generation/selection information. 
class ModelFileManager():
    """ Create full file paths of machine learning models. This class can be used for both training and inference.
    """
    def __init__(self ,modelInfoList = setting.myModelInfoList):
        """ set modelInfoList
        If there is no input related to modelInfoList, the information of the setting file is used.
        Therefore, setting.py file can be used to declare a general model information.

        :param modelInfoList: modelInfo for making full-ModelFilePaths
        :type modelInfoList: json dictionary

        example
            >>> myModelRootPath = ["c:","Users", "bunny", "Code_CLUST", "KETIToolDL","DL", "Models"]
            >>> myModelInfoList = {
                            "brits":{
                                "modelRootPath": myModelRootPath, 
                                "modelInfoPath": ["brits"],
                                "modelFileNames":['model.json', 'model.pth']
                    }
                }

        """ 
        self.modelInfoList = modelInfoList

    def getModelFilePath(self, trainDataPath, method):
        """ get fullModelFilePath
        Ths function makes fullModelFilePath list.
        TrainDataPath and other paths obtained by method can be used for creating fullModelFilePath.

        :param trainDataPath: It includes train data information to generate model path
        :type trainDataPath: list of str

        :param method: train method
        :type method: str

        example
            >>>  from KETIToolDL import modelInfo
            >>>  MI = modelInfo.ModelFileManager()
            >>>  trainDataPath =['DBName', 'MSName', 'columnName' ]
            >>>  trainMethod ='brits'
            >>>  modelFilePath = MI.getModelFilePath(trainDataPath, self.trainMethod)
        """ 
        modelInfo = self.modelInfoList[method]
        
        modelFullPath =modelInfo['modelRootPath']+modelInfo['modelInfoPath']+trainDataPath
        modelFolderPath=''
        for addfolder in modelFullPath:
            modelFolderPath = os.path.join(modelFolderPath, addfolder)

        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)
        
        modelFilePath=[]
        for i, model_name in enumerate(modelInfo['modelFileNames']):
            modelFilePath.append(os.path.join(modelFolderPath, model_name))
        return modelFilePath




