import pandas as pd
import os
# 


## 2. DataSelection
def getSavedIntegratedData(dataSaveMode, dataName, dataRoot, db_client = None):
    if dataSaveMode =='CSV':
        current = os.getcwd()
        print(current)
        fileName = os.path.join(current, dataRoot, dataName +'.csv')
        data = pd.read_csv(fileName, index_col='datetime', infer_datetime_format=True, parse_dates=['datetime'])

    elif dataSaveMode =='influx':

        db_name = dataRoot
        ms_name = dataName
        data = db_client.get_data(db_name, ms_name)
        
    return data
