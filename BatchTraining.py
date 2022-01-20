import os
import sys
from multiprocessing import freeze_support
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("../../../..")

from KETIPreDataIngestion.KETI_setting import influx_setting_KETI as ins
from KETIPreDataIngestion.data_influx import influx_Client
        
##########################################
model_method_list = ['brits']
model_purpose_list=['imputation']
model_purpose = model_purpose_list[0]
model_method = model_method_list[0]
##########################################
DBClient = influx_Client.influxClient(ins.CLUSTDataServer)

first ='2020-06-18'
last ='2020-06-19'
bind_params = {'end_time':last, 'start_time': first}
##########################################
model_rootDir = os.path.join('C:\\Users', 'bunny','Code_CLUST', 'DL', 'Models')
model_rootDir = os.path.join(model_rootDir, model_purpose, model_method)

if __name__ == '__main__':
    freeze_support()
    ###########################################
    #0. Define Trainer
    from KETIToolDL.Model.trainer import BritsTrainer
    ModelTrainer = BritsTrainer()

    #1-1. Define Influx Trainer - MS Data Batch
    db_name = 'air_indoor_요양원'
    ms_name = 'ICL1L2000017'
    from KETIToolDL.BatchTrainer.influxDB import InfluxTrainer
    trainer = InfluxTrainer(DBClient, model_method, model_rootDir)
    trainer.setTrainer(ModelTrainer)
    trainer.trainerForMS(db_name, ms_name, bind_params)

    ###########################################
    #1-2. Define Influx Trainer - DB Data Batch
    db_name = 'air_indoor_요양원'
    trainer = InfluxTrainer(DBClient, model_method, model_rootDir)
    trainer.setTrainer(ModelTrainer)
    trainer.trainerForDB(db_name, bind_params)

