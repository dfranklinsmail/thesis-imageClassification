from os import listdir
from os import mkdir
from os import path
import random
from shutil import copyfile

inputDir = '/Users/sircrashalot/Documents/school/thesis/proteins6New'
outputDir = '/Users/sircrashalot/Documents/school/thesis/proteins6NewSorted'

ignoreFilename = '.DS_Store'

class DataSorter():
    def main(self):
        directories = listdir(inputDir)
    
        for d in directories:
            if d != ignoreFilename:
                sourceDir = inputDir+'/'+d
                files = self.getFiles(sourceDir)
                self.saveData(sourceDir, outputDir+'/training', d, files[:1493])
                self.saveData(sourceDir, outputDir+'/validating', d, files[1493:1683])
                self.saveData(sourceDir, outputDir+'/testing', d, files[1683:1870])


    def getFiles(self, dirPath):
        files = listdir(dirPath)
        files.pop(0)
        random.shuffle(files)
        return files

    def saveData(self, sourceDir, destinationDir, folderName, files):
        
        saveDir = destinationDir+'/'+folderName
        print(saveDir)
        mkdir(saveDir)
        for f in files:
            copyfile(sourceDir+'/'+f, saveDir+'/'+f)


if __name__ == '__main__':
    dataSorter = DataSorter()
    dataSorter.main()
