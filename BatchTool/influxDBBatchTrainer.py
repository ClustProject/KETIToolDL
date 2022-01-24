import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

from KETIToolDL.BatchTool.batchTrainer import BatchTrainer

class InfluxDBBatch(BatchTrainer):
    def __init__(self, parameter, dbClient):
        super().__init__(parameter)
        self.DBClient = dbClient

    def trainerForDBMS(self, db_name, bind_params):
        self.MSList = self.DBClient.measurement_list(db_name)
        for ms_name in self.MSList:
            self.trainerForMS(db_name, ms_name, bind_params)

    def trainerForDBColumns(self, db_name, bind_params):
        self.MSList = self.DBClient.measurement_list(db_name)
        for ms_name in self.MSList:
            self.trainerForMSColumn(db_name, ms_name, bind_params)

    def trainerForMS(self, db_name, ms_name, bind_params):
        model_name_list=[db_name, ms_name]
        model_folderNameList = self.getModelFolderNameList(self.modelFullDirRootPath, model_name_list)

        df = self.DBClient.get_data_by_time(bind_params, db_name, ms_name)
        model_name = ms_name
        self.trainer.trainModel(df[[model_name]],  model_folderNameList, model_name, self.modelFileExtension)

    def trainerForMSColumn(self, db_name, ms_name, bind_params):
        model_name_list=[db_name, ms_name]
        model_folderNameList = self.getModelFolderNameList(self.modelFullDirRootPath, model_name_list)

        df = self.DBClient.get_data_by_time(bind_params, db_name, ms_name)

        for column_name in df.columns:
            model_name = column_name    
            self.trainer.trainModel(df[[model_name]],  model_folderNameList, model_name, self.modelFileExtension)

    