import csv
from matplotlib.pyplot import errorbar
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import statistics
import os

def readCsv(fileName):

    fields = []
    rows = []


    with open(fileName, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        fields = next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)

            # get total number of rows
        print("Total no. of rows: %d" % (csvreader.line_num))

    # printing the field names
    print('Field names are:' + ', '.join(field for field in fields))

    #  printing first 5 rows
    print('\nFirst 5 rows are:\n')
    for row in rows:
        print(row)



def readCsvAsDf(fileName):
    Candidates = list()
    Maxdepth = list()
    MinSamples = list()
    WindowSize = list()
    RemoveCanddates = list()
    useValidationSet = list()
    k = list()
    PercentageTrainingSet = list()
    useClustering = list()
    NumClusterMedoid = list()
    Time = list()

    with open(fileName, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        Percentage = list()
        Accuracy = list()
        i = 0
        # extracting each data row one by one
        for row in csvreader:
            if (i == 0):
                i = 1
                continue
            Candidates.append(row[0])
            Maxdepth.append(row[1])
            MinSamples.append(row[2])
            WindowSize.append(row[3])
            RemoveCanddates.append(row[4])
            k.append(row[5])
            useValidationSet.append(row[6])
            PercentageTrainingSet.append(row[7])
            useClustering.append(row[8])
            NumClusterMedoid.append(row[9])
            Accuracy.append(row[10])
            if (row[11] != None):
                Time.append(row[11])
            else:
                Time.append(0)
        # get total number of rows

        dfResults = pd.DataFrame(
            columns=['Candidates', 'MaxDepth', 'MinSamples', 'WindowSize', 'RemoveCanddates', 'k', 'useValidationSet'
                                                                                                   'PercentageTrainingSet',
                     'useClustering', 'NumClusterMedoid', 'Accuracy', 'Time'], index=range(csvreader.line_num - 1))


    dfResults['Candidates'] = Candidates
    dfResults['MaxDepth'] = Maxdepth
    dfResults['MinSamples'] = MinSamples
    dfResults['WindowSize'] = WindowSize
    dfResults['RemoveCandidates'] = RemoveCanddates
    dfResults['k'] = k
    dfResults['useValidationSet'] = useValidationSet
    dfResults['PercentageTrainingSet'] = PercentageTrainingSet
    dfResults['useClustering'] = useClustering
    dfResults['NumClusterMedoid'] = NumClusterMedoid
    dfResults['Accuracy'] = Accuracy
    dfResults['Time'] = Time

    dfResults['Accuracy'] = list(map(float, dfResults['Accuracy']))

    return dfResults

#Write per MAPIC
def WriteCsvMAPIC(fileName,row):
    fields = ['Dataset','Candidates', 'Max depth', 'Min samples', 'Window size', 'Remove candi', 'k', 'useValidationSet' ,'% Training set', 'useClustering','NumCluster(Medoids)' ,'Accuracy','PreprocessingTrainTime','TrainTime','PreprocessingTestTime','PredictionTime']
    #fields = ['Algorithm','Dataset','Accuracy','PreprocessingTrainTime','TrainTime','PreprocessingTestTime','TestTime','avgSSE','avgNumIterationKMeans']

    writeFileds=False
    if(os.path.isfile(fileName)==False):
        writeFileds=True

    # writing to csv file
    with open(fileName, 'a', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        if (writeFileds):
            csvwriter.writerow(fields)
        # writing the data rows
        csvwriter.writerow(row)


#Write per MAPIC
def WriteCsvShapeletAlgo(fileName,row):
    #fields = ['Dataset','Candidates', 'Max depth', 'Min samples', 'Window size', 'Remove candi', 'k', 'useValidationSet' ,'% Training set', 'useClustering','NumCluster(Medoids)' ,'Accuracy','Time']
    fields = ['Algorithm','Dataset','Accuracy','PreprocessingTrainTime','TrainTime','PreprocessingTestTime','TestTime']

    writeFileds=False
    if(os.path.isfile(fileName)==False):
        writeFileds=True

    # writing to csv file
    with open(fileName, 'a', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        if (writeFileds):
            csvwriter.writerow(fields)
        # writing the data rows
        csvwriter.writerow(row)


#Write per altri algoritmi
def WriteCsvComparison(fileName,row):
    fields = ['Algorithm', 'DatasetName', 'Accuracy', 'PreProcessingTrainTime' , 'TrainTime','PredictionTime']
    writeFileds=False
    if(os.path.isfile(fileName)==False):
        writeFileds=True

    # writing to csv file
    with open(fileName, 'a', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        if (writeFileds):
            csvwriter.writerow(fields)
        # writing the data rows
        csvwriter.writerow(row)








def buildTable(fileName,datasetName,query):

    dfResult=readCsvAsDf(fileName)

    print(dfResult)

    dfResult=dfResult[(dfResult['useValidationSet']=="False")]

    print(dfResult)

    #['Motifs','3','10','5','1','2','40']
    dfLocal = dfResult[(dfResult['Candidates'] == query[0])]
    dfLocal = dfLocal[(dfLocal['MaxDepth'] == query[1])]
    print('prinop')
    print(dfLocal)
    dfLocal = dfLocal[(dfLocal['MinSamples'] == query[2])]
    dfLocal = dfLocal[(dfLocal['WindowSize'] == query[3])]
    dfLocal = dfLocal[(dfLocal['RemoveCandidates'] == query[4])]
    dfLocal = dfLocal[(dfLocal['k'] == query[5])]
    dfLocal = dfLocal[(dfLocal['NumClusterMedoid'] == query[6])]

    print(dfLocal)

    accuracy = dfLocal['Accuracy'].values
    time = dfLocal['Time'].values


    accuracy=max(accuracy)
    time=min(time)

    row=[datasetName,accuracy,time]

    fields = ['NameDataset', 'Accuracy', 'Execution Time']
    writeFileds = False
    if (os.path.isfile(fileName) == False):
        writeFileds = True

    # writing to csv file
    with open(fileName, 'a', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        if (writeFileds):
            csvwriter.writerow(fields)
        # writing the data rows
        csvwriter.writerow(row)











