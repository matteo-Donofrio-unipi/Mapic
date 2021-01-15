from TestManager import executeMAPIC, buildTable, executeShapeletTransform, executeClassicDtree, \
    executeDecisionTreeStandard, executeKNN, executeLearningShapelet
from FileManager import WriteCsvComparison
from PlotLibrary import plotComparisonMultiple,plotTs, plotTestResults, plotComparisonSingle
def main():

    DatasetNames=["ArrowHead","BirdChicken","Coffee","Earthquakes", "ECG200",
                   "ECG5000","FaceFour","GunPoint","ItalyPowerDemand","OliveOil","PhalangesOutlinesCorrect",
                   "Strawberry","Trace","TwoLeadECG","Wafer","Wine","Worms","WormsTwoClass","Yoga"]

    useValidationSet = False
    usePercentageTrainingSet = True

    datasetName="ECG200"
    nameFile = datasetName + 'TestResults.csv'

    executeMAPIC(useValidationSet, usePercentageTrainingSet, datasetName, nameFile)

    #executeShapeletTransform(datasetName)

    #executeLearningShapelet(datasetName)

    #executeClassicDtree(datasetName) #con shapelet

    #executeKNN(datasetName)

    #executeDecisionTreeStandard(datasetName)


    #plotTs(datasetName)

    #plotTestResults(nameFile,datasetName) #prende in considerazione solo i risultati ottenuti su test set (senza validation)

    #fissato il primo su x, vario il secondo su y
    max=0 #take best accuracy and min time
    avg=1 #take avg time and accuracy
    min=-1 #take min time and accuracy


    #plotComparisonMultiple(nameFile,datasetName,'MaxDepth','Candidates',max)

    #plotComparisonSingle(nameFile,datasetName,'Candidates',max,UsePercentageTrainingSet=False)


if __name__ == "__main__":
    main()
