import os
import sys
from multiprocessing import freeze_support
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

from KETIPreDataIngestion.KETI_setting import influx_setting_KETI as ins
from KETIPreDataIngestion.data_influx import influx_Client

dataParameter = {
    "bind_params" :{'end_time':'2020-06-18 15:00:00', 'start_time': '2020-06-18 00:00:00'},
    "db_name":'air_indoor_요양원'
}
modelParameter= {
    "model_rootPath" :['DL', 'Models'],
    "model_method":'brits', 
    "model_fileName":['model.json', 'model.pth']
}
modelTrainParameter={

}

if __name__ == '__main__':
    freeze_support()
    DBClient = influx_Client.influxClient(ins.CLUSTDataServer)
    
    ##########################################
    #0. Define Trainer
    from KETIToolDL.TrainTool.trainer import BritsTrainer
    Brits = BritsTrainer()
    train_param = None
    Brits.setTrainParameter(modelTrainParameter)
  
    from KETIToolDL.BatchTool.influxDBBatchTrainer import InfluxDBBatch
    trainer = InfluxDBBatch(DBClient)
    trainer.setTrainMethod(Brits)
    trainer.setParameter(dataParameter, modelParameter)
    trainer.batchTrain()
    


