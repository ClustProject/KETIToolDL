import os
def setPathInfo(method, modelSetting, trainDataPath):
        PathInfo={}
        
        PathInfo['ModelRootPath'] = modelSetting.model_rootPath
        PathInfo['ModelInfoPath'] = modelSetting.modelParameterInfoList[method]["model_method"]
        PathInfo['TrainDataPath'] = trainDataPath
        PathInfo['ModelFileName'] = modelSetting.modelParameterInfoList[method]["model_fileName"]
        
        return PathInfo

def setModelFilesName(PathInfo):
    modelFolderpath =''
    for add_folder in PathInfo['ModelRootPath']:
        modelFolderpath = os.path.join(modelFolderpath, add_folder)
    for add_folder in PathInfo['ModelInfoPath']:
        modelFolderpath = os.path.join(modelFolderpath, add_folder)
    for add_folder in PathInfo['TrainDataPath']:
        modelFolderpath = os.path.join(modelFolderpath, add_folder)
        
    if not os.path.exists(modelFolderpath):
        os.makedirs(modelFolderpath)
    
    modelFilePath=[]
    for i, model_name in enumerate(PathInfo['ModelFileName']):
        modelFilePath.append(os.path.join(modelFolderpath, model_name))
    return modelFilePath