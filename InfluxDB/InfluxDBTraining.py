import os
class InfluxDBTraining():
    def __init__(self, dbClient, rootDir):
        self.DBClient = dbClient
        self.root_dir = rootDir

    def trainerForDB(self, db_name, bind_params):
        self.MSList = self.DBClient.measurement_list(self.db_name)
        for ms_name in self.MSList:
            self.trainerForMS(db_name, ms_name, bind_params)

    def trainerForMS(self, db_name, ms_name, bind_params):
        self.db_name = db_name
        self.ms_name = ms_name
        df = self.DBClient.get_data_by_time(bind_params, db_name, ms_name)
        self.trainerForColumnData(df)
        
    def trainerForColumnData(self, df):
        for column_name in df.columns:
            column_data = df[[column_name]]
            self.model_folder = self.setModelFolder()
            self.checkFolder(self.model_folder)
            self.setModelFileName(column_name)
            self.trainSaveModel(column_data)
            print("Model Saved")

    def checkFolder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder) 

    ## Abstract 
    def setModelFolder(self):
        model_folder = os.path.join(self.root_dir, self.db_name, self.ms_name) 
        return model_folder

    def setModelFileName(self, column_name):
        pass

    def trainSaveModel(self, df):
        pass