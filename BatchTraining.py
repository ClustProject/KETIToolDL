import os
import sys
import torch
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
##########################################
DBClient = influx_Client.influxClient(ins.CLUSTDataServer)
db_name = 'air_indoor_요양원'
ms_name = 'ICL1L2000017'
#first = DBClient.get_first_time(db_name, ms_name)
#last = DBClient.get_last_time(db_name, ms_name)
first ='2020-06-18'
last ='2020-06-19'
##########################################
bind_params = {'end_time':last, 'start_time': first}
model_purpose = model_purpose_list[0]
model_method = model_method_list[0]
model_rootDir = os.path.join('C:\\Users', 'bunny','Code_CLUST', 'DL', 'Models', model_purpose, model_method)
###########################################
# 1. Training Influx DB Data
from trainBySource import influxDBData
mode_list = ['MS_Training', 'DB_Training']
mode = mode_list[0] ## mode select
influxDBData.train(model_method, mode, DBClient, db_name, ms_name, bind_params, model_rootDir)