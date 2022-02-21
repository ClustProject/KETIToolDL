import os

from KETIToolDL import generalModelInfo 
def getmodelFilePath(trainDataPath, method):
    modelRootPath, modelInfoPath, modelFileNames =getModePathFromInfo(generalModelInfo, method)
    pathInfo = setPathInfo(modelRootPath, modelInfoPath, trainDataPath, modelFileNames)
    modelFilePath = setModelFilesName(pathInfo)
    return modelFilePath

def getModePathFromInfo(generalModelInfo, method):
    modelInfo = generalModelInfo.modelParameterInfoList[method]
    modelRootPath = modelInfo['modelRootPath']
    modelInfoPath = modelInfo['modelInfoPath']
    modelFileNames = modelInfo['modelFileNames']

    return modelRootPath, modelInfoPath, modelFileNames

def setPathInfo(modelRootPath, modelInfoPath, trainDataPath, modelFileNames):
    PathInfo={}
    PathInfo['modelRootPath']= modelRootPath
    PathInfo['modelInfoPath'] = modelInfoPath
    PathInfo['trainDataPath'] = trainDataPath
    PathInfo['modelFileNames'] = modelFileNames
    return PathInfo


def setModelFilesName(PathInfo):
    modelFolderpath =''
    for add_folder in PathInfo['modelRootPath']:
        modelFolderpath = os.path.join(modelFolderpath, add_folder)
    for add_folder in PathInfo['modelInfoPath']:
        modelFolderpath = os.path.join(modelFolderpath, add_folder)
    for add_folder in PathInfo['trainDataPath']:
        modelFolderpath = os.path.join(modelFolderpath, add_folder)
        
    if not os.path.exists(modelFolderpath):
        os.makedirs(modelFolderpath)
    
    modelFilePath=[]
    for i, model_name in enumerate(PathInfo['modelFileNames']):
        modelFilePath.append(os.path.join(modelFolderpath, model_name))
    return modelFilePath