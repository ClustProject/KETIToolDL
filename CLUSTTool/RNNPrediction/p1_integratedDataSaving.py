import os, sys
import json

sys.path.append("../../")
sys.path.append("../../..")

import setting

## 1. IntegratedDataSaving
#JH TODO 아래 코드에 대한 주석 작성
#JH TODO Influx Save Load 부분 작성 보완해야함


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

def getData(db_client, dataInfo, integration_freq_sec, processParam, startTime, endTime):
    from KETIPreDataSelection.data_selection.setSelectionParameter import makeIntDataInfoSet
    intDataInfo = makeIntDataInfoSet(dataInfo, startTime, endTime)
    def getIntegrationParam(integration_freq_sec):
        integration_param = {
            "granularity_sec":integration_freq_sec,
            "param":{},
            "method":"meta"
        }
        return integration_param

    integrationParam = getIntegrationParam(integration_freq_sec)
    
    from KETIPreDataIntegration.clustDataIntegration import ClustIntegration
    data = ClustIntegration().clustIntegrationFromInfluxSource(db_client, intDataInfo, processParam, integrationParam)
    
    return data

def saveData(data, dataDescriptionInfo, timeIntervalInfo, savemode='CSV'):
    if savemode =='influx':
        #TODO JH : InfluxSave
        #
        pass
    elif savemode == 'CSV' :
        # mode is CSV or others
        #File Save
        fileName = setting.csvDataFileRootDir+(dataDescriptionInfo+'_'+timeIntervalInfo)+'.csv'
        data.to_csv(fileName)
    else:
        # CSV, Influx 모두 저장?
        pass

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


def setMetaData(DataMeta, dataDescriptionInfo, timeIntervalInfo, processParam, dataInfo, integration_freq_sec,startTime, endTime, cleanParam, DataSaveMode):
    dataName = dataDescriptionInfo +'_'+ timeIntervalInfo 

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
    
    return DataMeta

def saveDataMeta(data, processParam, dataInfo, integration_freq_sec, cleanParam, DataSaveMode, startTime, endTime):
    from KETIPreDataTransformation.general_transformation.dataScaler import encodeHashStyle

    dataDescriptionInfo = encodeHashStyle(getListMerge([str(processParam), str(dataInfo), str(integration_freq_sec), cleanParam, DataSaveMode]))
    timeIntervalInfo = encodeHashStyle(getListMerge([startTime, endTime]))
    saveData(data, dataDescriptionInfo, timeIntervalInfo, DataSaveMode)

    # 2-4-3
    DataMeta = readJsonData(setting.DataMetaPath)
    updateMeta = setMetaData(DataMeta, dataDescriptionInfo, timeIntervalInfo, processParam, dataInfo, integration_freq_sec,startTime, endTime, cleanParam, DataSaveMode)
    writeJsonData(setting.DataMetaPath, updateMeta)

if __name__ == "__main__":
    
    #1-1
    dataInfo = [['farm_swine_air', 'HS1'], ['weather_outdoor_keti_clean', 'sangju'], ['life_additional_Info', 'trigonometicInfoByHours']]

    #1-2
    integration_freq_sec = 60 * 60 # 60분

    # 1-3 
    DataSaveMode='CSV' #or influx

    # 2 
    # 2-1 (Train)
    trainStartTime = "2020-11-25 00:00:00"
    trainEndTime ="2021-02-28 00:00:00"
    testStartTime ="2021-03-01 00:00:00"
    testEndTime ="2021-03-31 00:00:00"

    #startTime = trainStartTime
    #endTime = trainEndTime

    startTime = testStartTime
    endTime = testEndTime

    # 2-2
    #cleanParam ="Clean"
    cleanParam = "NoClean"

    # 2-3
    processParam = getProcessParam(cleanParam) 

    # 2-4
    data = getData(setting.db_client, dataInfo, integration_freq_sec, processParam, startTime, endTime)
    # 2-5
    saveDataMeta(data, processParam, dataInfo, integration_freq_sec, cleanParam, DataSaveMode, startTime, endTime)