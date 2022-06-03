# Modify this Path to make adaptive model path environment

#myModelRootPath = ["/Users", "jw_macmini", "CLUSTGit", "DL", "Models"]
#myModelRootPath = ["/Users", "bunnyjw", "Git", "DL", "Models"]
myModelRootPath = ['/home','keti', 'CLUST_KETI', 'Clust', 'KETIAppTestCode','JHTest', 'Models']
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
                "modelFileNames":['model_state_dict.pth']}
}
