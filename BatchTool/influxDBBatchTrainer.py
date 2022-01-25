import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

class InfluxDBBatch():
    def __init__(self, dbClient):
        self.DBClient = dbClient

    def setParameter(self, dataParameter, modelParameter):
        self.dataParameter = dataParameter
        self.modelParameter = modelParameter
    
    def setTrainMethod(self, trainMethod):
        self.trainer = trainMethod

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
        for column_name in df.columns: 
            PathInfo = self.setPathInfo(db_name, ms_name, column_name, bind_params)
            self.trainer.trainModel(df[[column_name]],  PathInfo)
    
    def setPathInfo(self, db_name, ms_name, column_name, bind_params):
        PathInfo={}
        PathInfo['ModelRootPath'] = self.modelParameter['model_rootPath']
        PathInfo['ModelInfoPath'] = [self.modelParameter['model_method']]
        PathInfo['TrainDataPath'] = [db_name, ms_name, column_name]#, str(bind_params)]
        PathInfo['ModelFileName'] = self.modelParameter['model_fileName']
        
        return PathInfo