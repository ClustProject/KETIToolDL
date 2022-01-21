import os
from batchTrainer import BatchTrainer

class InfluxDBBatch(BatchTrainer):
    def __init__(self, dbClient, model_rootDir):
        self.DBClient = dbClient
        self.model_rootDir = model_rootDir

    def setTrainer(self, trainer):
        self.trainer = trainer

    def trainerForDB(self, db_name, bind_params):
        self.MSList = self.DBClient.measurement_list(db_name)
        for ms_name in self.MSList:
            self.trainerForMS(db_name, ms_name, bind_params)

    def trainerForDBColumns(self, db_name, bind_params):
        self.MSList = self.DBClient.measurement_list(db_name)
        for ms_name in self.MSList:
            self.trainerForMSColumn(db_name, ms_name, bind_params)

    def trainerForMS(self, db_name, ms_name, bind_params):
        model_name_list=[db_name, ms_name]
        model_folder = self.getModelFolderName(self.model_rootDir, model_name_list)

        df = self.DBClient.get_data_by_time(bind_params, db_name, ms_name)
        model_name = ms_name
        self.trainer.trainModel(model_folder, model_name, df[[model_name]])

    def trainerForMSColumn(self, db_name, ms_name, bind_params):
        model_name_list=[db_name, ms_name]
        model_folder = self.getModelFolderName(self.model_rootDir, model_name_list)
        print(model_folder)
        df = self.DBClient.get_data_by_time(bind_params, db_name, ms_name)

        for column_name in df.columns:
            model_name = column_name    
            self.trainer.trainModel(model_folder, model_name, df[[model_name]])

    