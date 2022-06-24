# Modify this Path to make adaptive model path environment

#myModelRootPath = ["/Users", "jw_macmini", "CLUSTGit", "DL", "Models"]
#myModelRootPath = ["/Users", "bunnyjw", "Git", "DL", "Models"]
# myModelRootPath = ['/home','leezy', 'CLUST_KETI', 'KETIAppTestCode','JHTest', 'models']
# myModelRootPath = ['/home','leezy','CLUST_KETI','KETIAppTestCode','JWTest', 'RNNStyleModelTest','models']
# myModelRootPath = ['/home','leezy','CLUST_KETI','KETIAppDataServer','model_Inference','models']
# myModelRootPath = ['/programdrive','ProjectServer','KETIAppDataServer','model_Inference','models']

#myModelRootPath = ['/home','keti', 'CLUST_KETI', 'Clust', 'KETIAppTestCode','JHTest', 'Models']
myModelRootPath = ['/home','keti', 'CLUST_KETI', 'Clust', 'KETIAppTestCode','JWTest', 'RNNStyleModelTest','Models']
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
            "":{
                "modelRootPath": myModelRootPath, 
                "modelInfoPath": ["gru"],
                "modelFileNames":['model_state_dict.pth']}
}
