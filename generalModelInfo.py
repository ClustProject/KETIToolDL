# This File includes a general model generation/selection information. Define each model root path, method, and file Names to be generated

model_rootPath = ["/Users", "jw_macmini", "CLUSTGit", "DL", "Models"]

modelParameterInfoList = {
    "brits":{
        "modelRootPath": model_rootPath, 
        "modelInfoPath": ["brits"],
        "modelFileNames":['model.json', 'model.pth']
    }
}


import os
def getmodelFilePath(trainDataPath, method):
    modelInfo = modelParameterInfoList[method]
    print(modelInfo['modelRootPath'])
    
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

