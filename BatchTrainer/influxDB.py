import os
class InfluxTrainer():
    def __init__(self, dbClient, model_method, model_rootDir):
        self.DBClient = dbClient
        self.model_method = model_method
        self.model_rootDir = model_rootDir

    def setTrainer(self, trainer):
        self.trainer = trainer

    def trainerForDB(self, db_name, bind_params):
        self.MSList = self.DBClient.measurement_list(self.db_name)
        for ms_name in self.MSList:
            self.trainerForMS(db_name, ms_name, bind_params)

    def trainerForMS(self, db_name, ms_name, bind_params):
        self.db_name = db_name
        self.ms_name = ms_name
        df = self.DBClient.get_data_by_time(bind_params, db_name, ms_name)
        model_folder = self.setModelFolder()
        self.trainerForColumnData(df, model_folder)

    def trainerForColumnData(self, df, model_folder):
        for column_name in df.columns:
            model_name = column_name
            column_data = df[[model_name]]
            self.trainer.setBasicModelInfo(model_folder, model_name)
            self.trainer.trainData(column_data)

    def setModelFolder(self):
        model_folder = os.path.join(self.model_rootDir, self.db_name, self.ms_name) 
        return model_folder