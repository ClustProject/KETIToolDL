import os
import sys
from multiprocessing import freeze_support
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

from KETIPreDataIngestion.KETI_setting import influx_setting_KETI as ins
from KETIPreDataIngestion.data_influx import influx_Client
        
parameter = {
    "bind_params" :{'end_time':'2020-06-18 15:00:00', 'start_time': '2020-06-18 00:00:00'},
    "model_method":'brits', 
    "modelFileExtension":['.json', '.pth'],
    "rootDir": ['DL', 'Models'],
    "modelDirPath":["model_method"]
}

def getModelDirPath(parameter):
    modelDirPath = ''
    modelDirPath_Add = parameter['modelDirPath']

    for addPath in parameter['rootDir']:
        modelDirPath = os.path.join(modelDirPath, addPath)

    for addPath in modelDirPath_Add:
        addPathName = parameter[addPath]
        modelDirPath = os.path.join(modelDirPath, addPathName)
    print(modelDirPath)
    return modelDirPath

if __name__ == '__main__':
    freeze_support()
    DBClient = influx_Client.influxClient(ins.CLUSTDataServer)
    modelDirPath = getModelDirPath(parameter)
    
    ##########################################
    #0. Define Trainer
    from KETIToolDL.Model.trainer import BritsTrainer
    Brits = BritsTrainer()
    train_param = None
    Brits.setTrainParameter(train_param)

    #1-1. Define Influx Trainer - MS Data Batch
    db_name = 'air_indoor_요양원'
    ms_name = 'ICL1L2000017'
    from KETIToolDL.BatchTool.influxDBBatchTrainer import InfluxDBBatch
    trainer = InfluxDBBatch(DBClient, modelDirPath, parameter['modelFileExtension'])
    trainer.setTrainMethod(Brits)
    trainer.trainerForMSColumn(db_name, ms_name, parameter['bind_params'])
    ###########################################
    #1-2. Define Influx Trainer - DB Data Batch
    db_name = 'air_indoor_요양원'
    trainer = InfluxDBBatch(DBClient, modelDirPath, parameter['modelFileExtension'])
    trainer.setTrainMethod(Brits)
    trainer.trainerForDBColumns(db_name, parameter['bind_param'])

