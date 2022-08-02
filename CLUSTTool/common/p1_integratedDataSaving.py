import os, sys
import json

sys.path.append("../../")
sys.path.append("../../..")

## 1. IntegratedDataSaving
#JH TODO 아래 코드에 대한 주석 작성
#JH TODO Influx Save Load 부분 작성 보완해야함


def saveAndUpdateDataAndMeta(dataRoot, DataMetaPath, data, processParam, dataInfo, integration_freq_sec, cleanParam, DataSaveMode, startTime, endTime, db_client=None):
    from KETIPreDataTransformation.general_transformation.dataScaler import encodeHashStyle
    dataDescriptionInfo = encodeHashStyle(getListMerge([str(processParam), str(dataInfo), str(integration_freq_sec), cleanParam, DataSaveMode]))
    timeIntervalInfo = encodeHashStyle(getListMerge([startTime, endTime]))
    dataName = dataDescriptionInfo+'_'+timeIntervalInfo
    
    saveData(data, DataSaveMode, dataName, dataRoot, db_client)

    # Save Json Meta
    saveJsonMeta(DataMetaPath, dataName, processParam, dataInfo, integration_freq_sec,startTime, endTime, cleanParam, DataSaveMode)
    # Save Mongo Meta

def getProcessParam(cleanParam):
    if cleanParam =="Clean":
        refine_param = {
        "removeDuplication":{"flag":True},
        "staticFrequency":{"flag":True, "frequency":None}
        }
        CertainParam= {'flag': True}
        uncertainParam= {'flag': True, "param":{
                  "outlierDetectorConfig":[
                        {'algorithm': 'IQR', 'percentile':99 ,'alg_parameter': {'weight':100}}    
        ]}}
        outlier_param ={
            "certainErrorToNaN":CertainParam, 
            "unCertainErrorToNaN":uncertainParam
        }
        imputation_param = {
            "serialImputation":{
                "flag":False,
                "imputation_method":[{"min":0,"max":3,"method":"linear", "parameter":{}}],
                "totalNonNanRatio":80
            }
        }

    else:
        refine_param = {
            "removeDuplication":{"flag":False},
            "staticFrequency":{"flag":False, "frequency":None}
        }
        CertainParam= {'flag': False}
        uncertainParam= {'flag': False, "param":{}}
        outlier_param ={
            "certainErrorToNaN":CertainParam, 
            "unCertainErrorToNaN":uncertainParam
        }
        imputation_param = {
            "serialImputation":{
                "flag":False,
                "imputation_method":[],
                "totalNonNanRatio":80
            }
        }
        
    process_param = {'refine_param':refine_param, 'outlier_param':outlier_param, 'imputation_param':imputation_param}
    return process_param

def getIntegrationParam(integration_freq_sec):
    integration_param = {
        "granularity_sec":integration_freq_sec,
        "param":{},
        "method":"meta"
    }
    return integration_param

def getData(db_client, dataInfo, integration_freq_sec, processParam, startTime, endTime):
    from KETIPreDataSelection.data_selection.setSelectionParameter import makeIntDataInfoSet
    intDataInfo = makeIntDataInfoSet(dataInfo, startTime, endTime)


    integrationParam = getIntegrationParam(integration_freq_sec)
    
    from KETIPreDataIntegration.clustDataIntegration import ClustIntegration
    data = ClustIntegration().clustIntegrationFromInfluxSource(db_client, intDataInfo, processParam, integrationParam)
    
    return data


def getListMerge(infoList):
    MergedName=''
    for info in infoList:
        MergedName = MergedName+info+'_'
    return MergedName

def writeJsonData(jsonFilePath, Data):
    if os.path.isfile(jsonFilePath):
            pass
    else: 
        directory = os.path.dirname(jsonFilePath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(jsonFilePath, 'w') as f:
            data={}
            json.dump(data, f, indent=2)
            print("New json file is created from data.json file")
            
    with open(jsonFilePath, 'w') as outfile:
        outfile.write(json.dumps(Data))

def readJsonData(jsonFilePath):
    
    if os.path.isfile(jsonFilePath):
            pass
    else: 
        directory = os.path.dirname(jsonFilePath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(jsonFilePath, 'w') as f:
            data={}
            json.dump(data, f, indent=2)
            print("New json file is created from data.json file")

    if os.path.isfile(jsonFilePath):
        with open(jsonFilePath, 'r') as json_file:
            jsonData = json.load(json_file)
    return jsonData

def saveData(data, DataSaveMode, dataName, dataRoot, db_client=None):
    print(DataSaveMode)
    if DataSaveMode =='influx':
        db_name = dataRoot
        ms_name = dataName
        db_client.write_db(db_name, ms_name, data)
        db_client.close_db()

    elif DataSaveMode == 'CSV' :
        current = os.getcwd()
        print(current)

        if not os.path.isdir(current + "/" + dataRoot) :
            os.mkdir(current + "/" + dataRoot)
        """
        else :
            for file in os.scandir(current + "/" + dataRoot) :
                os.remove(file.path)
        """
        fileName = os.path.join(current, dataRoot, dataName +'.csv')

        data.to_csv(fileName)

def saveJsonMeta(DataMetaPath, dataName, processParam, dataInfo, integration_freq_sec,startTime, endTime, cleanParam, DataSaveMode):

    DataMeta = readJsonData(DataMetaPath)

    DataInfo ={}
    DataInfo ["startTime"] = startTime
    DataInfo ["endTime"] = endTime
    DataInfo ["dataInfo"] = dataInfo
    DataInfo ["processParam"] = processParam
    DataInfo ["integration_freq_sec"] = integration_freq_sec
    DataInfo ["cleanParam"] = cleanParam
    DataInfo ["DataSaveMode"] = DataSaveMode
    DataMeta[dataName]={}
    DataMeta[dataName]["integrationInfo"] = DataInfo
    writeJsonData(DataMetaPath, DataMeta)

