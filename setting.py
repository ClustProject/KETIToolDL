# Modify this Path to make adaptive model path environment

#myModelRootPath = ["/Users", "jw_macmini", "CLUSTGit", "DL", "Models"]
#myModelRootPath = ["/Users", "bunnyjw", "Git", "DL", "Models"]
# myModelRootPath = ['/home','leezy', 'CLUST_KETI', 'KETIAppTestCode','JHTest', 'models']
# myModelRootPath = ['/home','leezy','CLUST_KETI','KETIAppTestCode','JWTest', 'RNNStyleModelTest','models']
# myModelRootPath = ['/home','leezy','CLUST_KETI','KETIAppDataServer','model_Inference','models']
# myModelRootPath = ['/programdrive','ProjectServer','KETIAppDataServer','model_Inference','models']

#myModelRootPath = ['/home','keti', 'CLUST_KETI', 'Clust', 'KETIAppTestCode','JHTest', 'Models']
myModelRootPath = ['/home','keti', 'CLUST_KETI', 'Clust', 'KETIAppTestCode','MLModelTest','Models']
#myModelRootPath = ['/home', 'hwangjisoo', '바탕화면', 'Clust', 'KETIAppTestCode','JSTest', 'ClassificationTest','KUDataClassification-main','Models']
#myModelRootPath = ["c:","Users", "bunny", "Code_CLUST", "KETIToolDL","DL", "Models"]
myModelInfoList = {
            "brits":{
                "modelRootPath": myModelRootPath, 
                "modelInfoPath": ["brits"],
                "modelFileNames":['model.json', 'model.pth']},
            "lstm":{
                "modelRootPath": myModelRootPath, 
                "modelInfoPath": ["lstm"],
                "modelFileNames":['model_state_dict.pth']},
            "gru":{
                "modelRootPath": myModelRootPath, 
                "modelInfoPath": ["gru"],
                "modelFileNames":['model_state_dict.pth']},
            # TODO 아래 클래시피케이션 중복 아닌지 확인해야함.
            "LSTM_cf":{
                "modelRootPath": myModelRootPath, 
                "modelInfoPath": ["LSTM_cf"],
                "modelFileNames":['model.pt']},
            "GRU_cf":{
                "modelRootPath": myModelRootPath, 
                "modelInfoPath": ["GRU_cf"],
                "modelFileNames":['model.pt']},
            "CNN_1D_cf":{
                "modelRootPath": myModelRootPath, 
                "modelInfoPath": ["CNN_1D_cf"],
                "modelFileNames":['model.pt']},
            "LSTM_FCNs_cf":{
                "modelRootPath": myModelRootPath, 
                "modelInfoPath": ["LSTM_FCNs_cf"],
                "modelFileNames":['model.pt']},
            "FC_cf":{
                "modelRootPath": myModelRootPath, 
                "modelInfoPath": ["FC_cf"],
                "modelFileNames":['model.pt']},
            # Regression Model
            "LSTM_rg":{
                "modelRootPath": myModelRootPath, 
                "modelInfoPath": ["LSTM_rg"],
                "modelFileNames":['model.pt']},
            "GRU_rg":{
                "modelRootPath": myModelRootPath, 
                "modelInfoPath": ["GRU_rg"],
                "modelFileNames":['model.pt']},
            "CNN_1D_rg":{
                "modelRootPath": myModelRootPath, 
                "modelInfoPath": ["CNN_1D_rg"],
                "modelFileNames":['model.pt']},
            "LSTM_FCNs_rg":{
                "modelRootPath": myModelRootPath, 
                "modelInfoPath": ["LSTM_FCNs_rg"],
                "modelFileNames":['model.pt']},
            "FC_rg":{
                "modelRootPath": myModelRootPath, 
                "modelInfoPath": ["FC_rg"],
                "modelFileNames":['model.pt']}
}
