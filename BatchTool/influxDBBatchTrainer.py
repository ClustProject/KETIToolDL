import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

class InfluxDBBatch():
    def __init__(self, dbClient):
        self.DBClient = dbClient

    def setParameter(self, dataParameter, modelSetting):
        self.dataParameter = dataParameter
        self.modelSetting = modelSetting
    
    def setTrainMethod(self, trainMethod, method):
        self.trainer = trainMethod
        self.method = method

    def batchTrain(self):
        db_name = self.dataParameter['db_name']
        #MSColumn
        if "ms_name" in self.dataParameter:
            self.trainerForMSColumn()
        #DBMSColumn
        else:
            msList = self.DBClient.measurement_list(db_name)
            for ms_name in msList:
                self.dataParameter['ms_name'] = ms_name
                self.trainerForMSColumn()

    def trainerForMSColumn(self):
        bind_params = self.dataParameter['bind_params']
        ms_name = self.dataParameter['ms_name']
        db_name = self.dataParameter['db_name']
        df = self.DBClient.get_data_by_time(bind_params, db_name, ms_name)
        print(df)
        for column_name in df.columns: 
            trainDataPath = [db_name, ms_name, column_name]#, str(bind_params)]
            
            from KETIToolDL.ModelTool import modelFileManager
            PathInfo = modelFileManager.setPathInfo(self.method, self.modelSetting, trainDataPath)
            modelFilePath = modelFileManager.setModelFilesName(PathInfo)
            self.trainer.trainModel(df[[column_name]],  modelFilePath)
    
    