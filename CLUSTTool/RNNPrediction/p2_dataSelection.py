import pandas as pd
import setting

# 
import p1_integratedDataSaving as p1
## 2. DataSelection
def getSavedIntegratedData(dataSaveMode, dataName):
    if dataSaveMode =='CSV':
        FileName = setting.csvDataFileRootDir + dataName + '.csv'
        data = pd.read_csv(FileName, index_col='datetime', infer_datetime_format=True, parse_dates=['datetime'])

    elif dataSaveMode =='influx':
        #2-2 JH TODO influx에서 DataName으로 데이터를 읽을 수 있도록 함
        #dataName을 _로 나누면 DB 이름과 MS이름으로 나뉘어짐
        db_name = 'ml_data_integration'
        ms_name = dataName
        data = setting.db_client.get_data(db_name, ms_name)
        
    return data

if __name__ == "__main__":
    # 1
    DataMeta = p1.readJsonData(setting.DataMetaPath)
    dataList =  list(DataMeta.keys())

    # 2
    dataName = dataList[0]
    dataSaveMode = DataMeta[dataName]["integrationInfo"]["DataSaveMode"]
    integration_freq_sec = DataMeta[dataName]["integrationInfo"]["integration_freq_sec"]
    # 3
    data = getSavedIntegratedData(dataSaveMode, dataName)
    print(data.head())
    
