
import os
import pandas as pd
import matplotlib.pyplot as plt

#파일 유틸리티 
class FileUtils:
    
   
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.img_dir = os.path.join(self.base_dir, '..', 'static', 'images')
        

    # 폴더가 존재하지 않는다면 생성하기
    def Path(self):
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        return self.base_dir, self.img_dir

    #파일 Path 
    def FilePathName(self, fileName):
        return os.path.join(self.img_dir, fileName)

    # csv파일 읽어들이기
    def getCsv(self, fName, encoding='utf-8', fileType='csv', sep=';'):
        file_path = os.path.join(os.path.dirname(__file__), '..', fileType, fName)
        file_path = os.path.normpath(file_path)  # 경로 정규화
        return pd.read_csv(file_path, encoding=encoding, sep=sep)

    #차트 이미지로 만들기
    def savePngToPath(self,filename,closeFlag):
        # filename = self.img_dir +filename
        
        print("=============== filename ==============")
        print({filename})
        print("=============== filename ==============")
        plt.savefig(filename)
        if closeFlag == True:
            plt.close()
        
        return f'/static/images/{filename}'
        