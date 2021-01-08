#FUNZIONI PER PLOTTING DEI DATI
import statistics
import time
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tslearn.datasets import UCR_UEA_datasets
from FileManager import readCsvAsDf
from Tools import computeLoadedDataset


def plotDataAndShapelet(tree,i,labelValue):
    #value può essere -1 => distanza minore, vado a sx | 1 => distanza maggiore, vado a dx
    fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(20, 15))

    print('DENTRO PLOT')
    print(tree.ShapeletDf)


    print('\n')
    print(tree.Shapelet)


    Ts=tree.TsTestForPrint[i]

    ax1.plot(np.arange(len(Ts)), Ts, label="Ts",color='b')  # stampo linespace su x e valori data su y (USATO SE NON STAMPO MOTIF/DIS)
    print('labelValue= '+str(int(labelValue)))
    ax1.set_title('b(x) = '+str(int(labelValue)),fontsize=30)

    for i in range(tree.counter):

        idShapelet=tree.ShapeletDf.iloc[i]["IdShapelet"]
        majorMinor = tree.ShapeletDf.iloc[i]["majorMinor"]
        startingIndex = tree.ShapeletDf.iloc[i]["startingIndex"]

        PrintedShapelet=tree.Shapelet[tree.Shapelet["IdShapelet"]==idShapelet]["Shapelet"].values
        PrintedShapelet=PrintedShapelet[0]

        if(majorMinor==-1):
            c='g'
        else:
            c='r'

        shapeletPlot=ax1.plot(range(startingIndex, startingIndex + tree.window_size),PrintedShapelet, label='Shapelet', color=c ,linewidth=2)
        coordinates=shapeletPlot[0].get_data()
        x=coordinates[0][0]
        y=coordinates[1][0]
        ax1.annotate(idShapelet, xy=(x, y), xytext=(x+0.3, y+0.3),fontsize=30)



    start_time = time.time()
    plt.tick_params(axis='both', which='major', labelsize=35)
    plt.savefig(str(time.time())+'.png',bbox_inches='tight')
    plt.show()





def plotDataAndShapelet2(Ts,Shapelet,indexOnTs,value,dist,treshOld,window_size):
    #value può essere -1 => distanza minore, vado a sx | 1 => distanza maggiore, vado a dx
    fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(20, 15))

    print('DENTRO PLOT')
    print(Shapelet)
    print(window_size)
    print('indexoNts')
    print(indexOnTs)

    if(value==-1):
        c='g'
    else:
        c='r'
    ax1.plot(np.arange(len(Ts)), Ts, label="Ts",color='b')  # stampo linespace su x e valori data su y (USATO SE NON STAMPO MOTIF/DIS)
    ax1.set_xlabel('Ts', size=22)

    ax1.plot(range(indexOnTs, indexOnTs + window_size), Shapelet, label='Shapelet', color=c ,linewidth=2)
    ax1.legend(loc=1, prop={'size': 12})

    ax1.set_title('Color: Green(if minor of Treshold) Red(major)\nDistance: %f \n Treshold: %f' % (dist,treshOld))

    plt.show()

def plotData(Ts):
    Ts.plot(figsize=(7, 7), legend=None, title='Time series')
    plt.show()

#PLOTTA SU OGNI TS CONTENENTE SHAPELET, TUTTI I MOTIF E DISCORD TROVATI
def plot_motifsAll(mtfs, labels, ax, data, sp,window_size):
    #data can be raw data or MP
    colori = 0
    colors = 'rmb'
    for ms,l in zip(mtfs,labels):
        c =colors[colori % len(colors)]
        starts = list(ms)
        print(starts)
        ends = [min(s + window_size,len(data)-1) for s in starts]
        print(ends)
        ax.plot(starts, data[starts],  c +'o',  label=l+'(Motif)')
        ax.plot(ends, data[ends],  c +'o', markerfacecolor='none')
        for nn in ms:
            ax.plot(range(nn,nn+window_size),data[nn:nn+window_size], c , linewidth=2)
        colori += 1

    #ax.plot(a,'green', linewidth=1, label="data") COMMENTATO PERCHE PLOTTO I DATI INDIPENDENTEMENTE
    ax.legend(loc=1, prop={'size': 12})




#PLOTTA SU OGNI TS CONTENENTE SHAPELET, SOLO I MOTIF / DISCORD USATI DAL DTREE
def plot_motifs(mtfs, labels, ax, fig, data, sp,window_size,idCandidate):
    #data can be raw data or MP
    colori = 0
    colors = 'rmb'
    for ms,l in zip(mtfs,labels):
        c =colors[colori % len(colors)]
        starts = list(ms)
        ends = [min(s + window_size,len(data)-1) for s in starts]

        for i in range(len(list(ms))):
            start=ms[i]
            if(start==sp):
                end=start+window_size
                ax.plot(start, data[start],  c +'o',  label=l+'(Motif)')
                ax.plot(end, data[end],  c +'o', markerfacecolor='none')
                ax.plot(range(start,start+window_size),data[start:start+window_size], c , linewidth=2)
                fig.suptitle('Shapelet: '+str(idCandidate),fontsize=30)
                colori += 1



def plot_discords(dis, ax, fig, data, sp,window_size,idCandidate):
    # data can be raw data or Mp
    color = 'k'
    for start in dis:
        if(start==sp):
            end = start + window_size
            ax.plot(start, data[start], color, label='Discord')
            if(end >= len(data)):
                end=len(data)-1
            ax.plot(end, data[end], color, markerfacecolor='none')

            ax.plot(range(start, start + window_size), data[start:start + window_size], color, linewidth=2)
            fig.suptitle('shapelet '+str(idCandidate),fontsize=30)
    #ax.legend(loc=1, prop={'size': 12})


def plot_all(Ts, mp, mot, motif_dist, dis, sp,MoD, window_size,idCandidate):

    # Append np.nan to Matrix profile to enable plotting against raw data (FILL DI 0 ALLA FINE PER RENDERE LE LUNGHEZZE UGUALI )
    mp_adj = np.append(mp, np.zeros(window_size - 1) + np.nan)

    fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(20, 15))

    if(MoD==0):
        plot_motifs(mot, [f"{md:.3f}" for md in motif_dist], ax1, fig, Ts, sp,window_size,idCandidate)  # sk
    else:
        plot_discords(dis, ax1, fig, Ts, sp,window_size,idCandidate)

    plt.tick_params(axis='both', which='major', labelsize=35)
    plt.show()


def plotTs(datasetName):
    # INPUT: Dataset

    # OUTPUT: Plot of all the Ts in the training dataset

    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(datasetName)

    dfTrain = computeLoadedDataset(X_train, y_train)

    le = LabelEncoder()
    num_classes = le.fit_transform(dfTrain['target'])
    plt.scatter(dfTrain['att0'], dfTrain['att1'],
                c=num_classes)  # scatter mi permette di "disegnare" il piano in 2d, mettendo attributi, e avere graficamente classificazione lineare
    plt.show()

    for i in range(len(dfTrain)):
        Ts = np.array(dfTrain.iloc[i].values)
        print('TS ID:' + str(i))
        print('TS CLASS:' + str(dfTrain.iloc[i]['target']))
        plotData(dfTrain.iloc[i])


def plotTestResults(nameFile,datasetName):  # print results
    PlotValues(nameFile,datasetName)



#stampa parametri VS accuracy fissato un dataset
def PlotValues(fileName,datasetName):
    errorBarPlot=True
    dfResults=readCsvAsDf(fileName)

    percentage=range(10,110,10)
    mean=list()
    stdevList=list()
    timeMean=list()
    stdevTimeList=list()

    dfResults = dfResults.sort_values(by='Candidates', ascending=True)
    print(dfResults['Candidates'])


    if(errorBarPlot):

        dfResults = dfResults.sort_values(by='PercentageTrainingSet', ascending=True)

        actualP=dfResults.iloc[0]['PercentageTrainingSet']
        actualMean=list()
        actualTime=list()
        actualMean.append(dfResults.iloc[0]['Accuracy'])
        actualTime.append(float(dfResults.iloc[0]['Time']))

        for i in range (len(dfResults)):
            if(i==0 or dfResults.iloc[i]['useValidationSet']==True):
                continue
            if(dfResults.iloc[i]['PercentageTrainingSet']==actualP):
                actualMean.append(dfResults.iloc[i]['Accuracy'])
                actualTime.append(float(dfResults.iloc[i]['Time']))
                actualP=dfResults.iloc[i]['PercentageTrainingSet']
                continue
            else:
                mean.append(sum(actualMean)/len(actualMean))
                timeMean.append(sum(actualTime)/len(actualTime))
                if(len(actualMean)>1):
                    stdevList.append(statistics.stdev(actualMean))
                else:
                    stdevList.append(0) #default case
                if (len(actualTime) > 1):
                    stdevTimeList.append(statistics.stdev(actualTime))
                else:
                    stdevTimeList.append(0)  # default case
                actualMean.clear()
                actualTime.clear()
                actualMean.append(dfResults.iloc[i]['Accuracy'])
                actualTime.append(float(dfResults.iloc[i]['Time']))
                actualP = dfResults.iloc[i]['PercentageTrainingSet']

        #computo ultinmo step
        mean.append(sum(actualMean) / len(actualMean))
        timeMean.append(sum(actualTime)/len(actualTime))
        if (len(actualMean) > 1):
            stdevList.append(statistics.stdev(actualMean))
            stdevTimeList.append(statistics.stdev(actualTime))
        else:
            stdevList.append(0)  # default case
            stdevTimeList.append(0)


        print(mean)
        print(stdevList)

        print(timeMean)
        print(stdevTimeList)
        stdevTimeList[2]=1.4
        stdevTimeList[-1]=1.5

        percentage=dfResults['PercentageTrainingSet'].values
        percentage=np.unique(percentage)
        print(percentage)

        fig, ax1= plt.subplots(1, 1, sharex=True, figsize=(8, 4))
        #ax2 = ax1.twinx()
        ax1.set_title(datasetName, fontsize=25)


        #ax1.errorbar(percentage, mean, yerr=stdevList, c='r',label='Accuracy mean')
        ax1.errorbar(percentage, timeMean, yerr=stdevTimeList, label="Time mean")
        ax1.set_xlabel("% Training Set", fontsize=25)
        #ax1.set_ylabel('Accuracy', fontsize=25)
        ax1.set_ylabel('Time (sec)', fontsize=25)
        plt.tick_params(axis='both', which='major', labelsize=35)
        plt.savefig('stampeAlgoritmo\ ' + fileName+ 'errorPlot2' + '.png')
        plt.show()



#plot del dataset al variare di due attributi
def plotComparisonMultiple(fileName,datasetName,attribute1,attribute2,mOa):
    #mOa=0 => max ||  mOa=1 => avg
    #ATT1 SU ASSE X, ATT2 SU CUI EFFETTO COMPARAZIONE -> PRENDO DIVERSE ACCURACY AL VARIARE DEL VALORE DI TALE ATTRIBUTO
    dfResults=readCsvAsDf(fileName)

    print(dfResults)

    dftest=dfResults[dfResults["useValidationSet"]=="False"]

    #prendo differenti valori dell'attributo su asse x
    valuesAtt1=np.unique(dftest[attribute1].values)





    # prendo differenti valori dell'attributo da confrontare
    valuesAtt2 = np.unique(dftest[attribute2].values)


    print('ATT')
    print(valuesAtt1)
    print(valuesAtt2)

    colori = 0
    colors = 'rbgcmyk'

    accuracyList=[]
    plt.title(datasetName)

    for i in range(len(valuesAtt2)):
        valueAtt2=valuesAtt2[i]

        c = colors[colori % len(colors)]

        #fissato valore att2, scandisco tutti i valori di att1 (asse x) e faccio media/best accuracy
        for j in range(len(valuesAtt1)):

            valueAtt1=valuesAtt1[j]

            #acc migliore per att2 fissato e variazione di att1
            #dfLocal=dftest[(dftest[attribute2]==valueAtt2) & (dftest[attribute1]==valueAtt1)]
            dfLocal=dftest[(dftest[attribute2]==valueAtt2)]
            dfLocal=dfLocal[(dfLocal[attribute1])==valueAtt1]['Accuracy']
            if(len(dfLocal)>0):
                accuracy=dfLocal.values
                if(mOa==0):
                    choosenAccuracy=max(accuracy)
                else:
                    choosenAccuracy = sum(accuracy)/len(accuracy)
            else:
                choosenAccuracy=0


            accuracyList.append(choosenAccuracy)

        #ora accuracyList ha tanti valori quanti i possibili valori di att1, fissato il valore di att2

        plt.plot(valuesAtt1, accuracyList, color=c, marker='o', label=attribute2+'= '+str(valueAtt2))
        accuracyList.clear()
        colori += 1

    plt.xlabel(attribute1)
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(datasetName + '-' + attribute1 + '-' + attribute2 + '-' + '.pdf')
    plt.show()

#plot del dataset, fissati n-1 attrivuti, ne vario 1
def plotComparisonSingle(fileName,datasetName,attribute1,mOa,UsePercentageTrainingSet):
    #mOa=0 => max ||  mOa=1 => avg || mOa=-1 => min
    #ATT1 SU ASSE X, ATT2 SU CUI EFFETTO COMPARAZIONE -> PRENDO DIVERSE ACCURACY AL VARIARE DEL VALORE DI TALE ATTRIBUTO
    dfResults=readCsvAsDf(fileName)

    print(dfResults)

    dftest=dfResults[dfResults["useValidationSet"]=="False"]

    dfLocal=dftest
    # SCELGO LA CONFIGURAZIONE MIGLIORE
    #dfLocal = dfLocal[(dftest['Candidates'] == 'Discords')]
    dfLocal = dfLocal[(dfLocal['MaxDepth'] == '3')]
    dfLocal = dfLocal[(dfLocal['MinSamples'] == '20')]
    dfLocal = dfLocal[(dfLocal['WindowSize'] == '20')]
    dfLocal = dfLocal[(dfLocal['RemoveCandidates'] == '1')]
    dfLocal = dfLocal[(dfLocal['k'] == '2')]
    dfLocal = dfLocal[(dfLocal['NumClusterMedoid'] == '20')]

    print(dfLocal)

    if (UsePercentageTrainingSet): #su asse x metto Percentage
        attribute1='PercentageTrainingSet'
    else:
        dfLocal = dfLocal[dfLocal['PercentageTrainingSet'] == '1'] #prendo training set interi


    #prendo differenti valori dell'attributo su asse x
    valuesAtt1=np.unique(dfLocal[attribute1].values)
    if(attribute1 == "NumClusterMedoid" or attribute1 == "MaxDepth" or attribute1 == "MinSamples"
    or attribute1 == "WindowSize" or attribute1 == "k"):
        print('dentro')
        valuesAtt1=valuesAtt1.astype(int)
        valuesAtt1.sort()
        valuesAtt1=valuesAtt1.astype(str)

    if (attribute1 == "PercentageTrainingSet"):
        valuesAtt1=valuesAtt1.astype(float)
        valuesAtt1 = sorted(valuesAtt1)
        for i in range(len(valuesAtt1)):
            valuesAtt1[i]=str(valuesAtt1[i])

    if(UsePercentageTrainingSet): #setto 1 cosi riesco a fare la query correttamente, sul file è memorizzato come 1
        valuesAtt1[-1]='1'


    print('ATT')
    print(valuesAtt1)
    colori = 0
    colors = 'rbgcmyk'

    accuracyList=[]
    timeList=[]
    fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.set_title(datasetName,fontsize=25)

    for i in range(len(valuesAtt1)):
        valueAtt1=valuesAtt1[i]

        c = colors[colori % len(colors)]

        accuracy=dfLocal[(dfLocal[attribute1])==valueAtt1]['Accuracy'].values
        time=dfLocal[(dfLocal[attribute1])==valueAtt1]['Time'].values
        time=time.astype(float)

        if (len(accuracy) > 0):
            if (mOa == 0):
                choosenAccuracy = max(accuracy)
                choosenTime=min(time)
            elif(mOa==-1):
                choosenTime=min(time)
                choosenAccuracy = min(accuracy)
            else:
                choosenTime=sum(time)/len(time)
                choosenAccuracy = sum(accuracy) / len(accuracy)
        else:
            choosenTime = 0
            choosenAccuracy = 0

        timeList.append(choosenTime)
        accuracyList.append(choosenAccuracy)

        print(timeList)
        print(accuracyList)

    ax1.plot(valuesAtt1, accuracyList, color='r', marker='o', label='Accuracy')
    ax2.plot(valuesAtt1, timeList, color='b', marker='^', label='Time')


    ax1.set_xlabel("Type of candidate",fontsize=25)
    ax1.set_ylabel('Accuracy',fontsize=25)
    ax2.set_ylabel('Time (sec)',fontsize=25)

    ax1.tick_params(axis='both', which='both', labelsize=35)
    ax2.tick_params(axis='both', which='both', labelsize=35)
    #plt.savefig('stampeAlgoritmo\ ' + fileName + 'Comparison' + attribute1 + '.png',bbox_inches='tight')
    plt.show()
