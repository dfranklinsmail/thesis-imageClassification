from os import listdir
from os import mkdir
from os import path
import random
from shutil import copyfile

inputDir = '/Users/sircrashalot/Documents/school/thesis/proteinsNew25PDB'
outputDir = '/Users/sircrashalot/Documents/school/thesis/proteinsNew25PDBSorted'
pdbFilePath = './25PDB.csv'

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

        uniqueList = []
        for filename in files:
            fn = filename[:-5]
            if fn not in uniqueList:
                uniqueList.append(fn)

        random.shuffle(uniqueList)

        
        sortedFiles = []
        for name in uniqueList:
            newName = name+'1.png'
            if newName in files:
               sortedFiles.append(newName)
            newName = name+'2.png'
            if newName in files:
               sortedFiles.append(newName)
            newName = name+'3.png'
            if newName in files:
               sortedFiles.append(newName)
            newName = name+'4.png'
            if newName in files:
               sortedFiles.append(newName)
            newName = name+'5.png'
            if newName in files:
               sortedFiles.append(newName)
            newName = name+'6.png'
            if newName in files:
               sortedFiles.append(newName)

        print('the sorted files are')
        print(sortedFiles)
        return sortedFiles

    def saveData(self, sourceDir, destinationDir, folderName, files):
        
        saveDir = destinationDir+'/'+folderName
        print(saveDir)
        mkdir(saveDir)
        for f in files:
            copyfile(sourceDir+'/'+f, saveDir+'/'+f)        

if __name__ == '__main__':
    dataSorter = DataSorter()
    dataSorter.main()
