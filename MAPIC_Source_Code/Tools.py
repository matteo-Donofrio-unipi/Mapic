import pandas as pd
import numpy as np
from matrixprofile import *
from matrixprofile.discords import discords
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import time
from scipy.spatial.distance import euclidean
from kMedoidsSearch import runKMedoids


# Library containing functions for:
# - the data preprocessing aimed at the generation of the dataset for the training phase
# - computation and optimisaztion of data structures during the fitting phase



def computeLoadedDataset(X, y):

    #INPUT: Raw dataset X (time series) and theirs labels y

    #OUTPUT: Structured single dataset (X,y)

    columnsList = np.arange(len(X[0]))
    columnsList2 = list()
    prefix = 'att'

    #Generates as many columns attributes as many values in the time series
    for i in columnsList:
        columnsList2.append(prefix + str(i))
    columnsList2.append('target')
    columnsList2.append('TsIndex')
    dataset = pd.DataFrame(columns=columnsList2, index=range(0, len(X)))

    #each rows of dataset contains a time series and its label (as last value)
    for i in range(len(X)):
        l1 = list()
        record = X[i]
        for j in range(len(X[i])):
            l1.append(record[j][0])
        l1.append(y[i])
        l1.append(i)
        dataset.iloc[i] = l1

    return dataset


def retrieve_all(tree,Ts,window_size,k):  # fornita la Ts calcola e restituisce mp, motifs, motifs_distances e discords

    #INPUT: Decision tree, Time series, sliding window size, #of motifs and discords extracted from each Ts

    #OUTPUT: Matrix Profile and the top K motifs and discords extracted from the Ts

    dfMP = pd.DataFrame(Ts).astype(float)
    dis=[]

    # IF the Ts contains almost all equal values, the MP Stomp function arises problems, so the MP its computed with the naive function

    if(tree.warningDetected==True):
        mp, mpi = matrixProfile.naiveMP(dfMP[0].values, window_size)

    else:
        mp, mpi = matrixProfile.stomp(dfMP[0].values, window_size)

    #check if the MP Stomp function arises problems
    if(np.isnan(mp).any() or np.isinf(mp).any()):
        tree.warningDetected=True
        print('switch to ComputeMpAndMpi')
        mp, mpi = matrixProfile.naiveMP(dfMP[0].values, window_size)


    # Tuple needed to extract Motifs
    tupla = mp, mpi

    mot, motif_dist = motifs.motifs(dfMP[0].values, tupla, k)

    #check if the MP Stomp function arises problems
    if(sum(mp)!=0):
        dis = discords(mp, window_size, k)
    # else impossible extract other discords (maybe windowSize too large)

    tupla = mp, mot, motif_dist, dis
    return tupla




#Our function to compute the MP and the MPI
def ComputeMpAndMpi(Ts, window_size):

    #INPUT: Time series, sliding window size

    #OUTPUT: MP and MPI

    if window_size >= len(Ts) or window_size < 2:
        raise ValueError('Window_size not supported')

    Ts = Ts.astype(float)
    lenTs = len(Ts)
    mp = list()
    mpi = list()

    for i in range(lenTs):

        bestDist = 1000
        bestIdx = 0

        if (i + window_size > lenTs):
            break
        else:
            subSeq = Ts[i:i + window_size]

            for j in range(lenTs):
                if(i==j):
                    continue
                if (j + window_size > lenTs):
                    break
                else:
                    subSeqToCompute = Ts[j:j + window_size]
                    dist=euclidean(subSeq,subSeqToCompute)

                    if (dist > 0 and dist < bestDist):
                        bestDist = dist
                        bestIdx = j  # starting index of founded closest subseq

            mp.append(bestDist)
            mpi.append(bestIdx)

    return mp, mpi

#CALCOLA DP TRA SUBSEQ A CONTENUTA IN TsContainigSubSeq E TUTTE LE SUBSEQ B CONTENUTE IN TsToCompare

#Our function to compute the Distance Profile
def ComputeDp(TsContainigSubSeq, indexStartigPosition, window_size, TsToCompare=None):

    #INPUT: Ts containing the subsequence (Query), index of the subsequence within Ts, Sliding window size, Ts whose distance has to be computed
    #TsToCompare=None if the Ts has to compute the distance with itself

    #OUTPUT: Distance profile btw the subsequence and the Ts

    TsContainigSubSeq = TsContainigSubSeq.astype(float)
    ComparingWithItSelf = False

    if (TsToCompare is None):
        ComparingWithItSelf = True
        TsToCompare = TsContainigSubSeq
    else:
        TsToCompare = TsToCompare.astype(float)

    if window_size >= len(TsToCompare) or window_size < 2:
        raise ValueError('Window_size not supported')

    #Query
    subSeq = TsContainigSubSeq[indexStartigPosition:indexStartigPosition + window_size]

    lenTs = len(TsToCompare)
    dp = list()

    for i in range(lenTs):

        if (i + window_size > lenTs):
            break
        elif (i == indexStartigPosition and ComparingWithItSelf == True):
            continue
        else:
            # subSeq generated by the sliding window
            subSeqToCompute = TsToCompare[i:i + window_size]
            dist = euclidean(subSeq, subSeqToCompute)
            dp.append(dist)

    return dp




# genero poi struttura contenente gli indici di partenza di tutti i candidati

#Each motif is identified by at least 2 starting positions, candidateFilter takes one starting position for each motifs
def candidateFilter(CandidatesList):

    #INPUT: A Dataframe containing in each row, the starting index positions of motifs and discrods extracted from a Ts

    #OUTPUT: A Dataframe containing in each row, only one starting index position for each motif and discord
    #        The overall number of motifs and discord extracted

    counterNumberMotif = 0
    counterNumberDiscord = 0
    l2 = np.array([])
    for i in range(len(CandidatesList['Motif'])):  # per ogni entry (per ogni record)
        numMotif = len(CandidatesList['Motif'].iloc[i])
        numDiscord = len(CandidatesList['Discord'].iloc[i])
        counterNumberDiscord += numDiscord
        for j in range(numMotif):  # for each motifs starting index position list
            l1 = CandidatesList['Motif'].iloc[i]  # motifs list
            l2 = np.append(l2, l1[j][0])  # takes the first starting position
            counterNumberMotif += 1

        CandidatesList['Motif'].iloc[i] = l2
        l2 = np.array([])

    return CandidatesList, counterNumberMotif, counterNumberDiscord



#conto dopo aver selezionato i medoidi, il num di motif e discords
def countNumberOfCandidates(CandidatesListTrain):

    numMotifs=0
    numDiscords=0

    numMotifs=len(CandidatesListTrain[CandidatesListTrain['M/D']==0])
    numDiscords = len(CandidatesListTrain[CandidatesListTrain['M/D'] == 1])

    return numMotifs,numDiscords





# Generates a boolean list of the same lentgh of the motifs and discords extracted from all Ts in the whole dataset
# in order to take trace of the alreay used candidates
def buildCandidatesUsedList(CandidatesList):

    #INPUT: List of the motifs and discords extracted from all Ts in the whole dataset

    #OUTPUT: List of as many boolean values as many motifs and discords extracted

    CandidatesUsedList = pd.DataFrame(columns=['IdCandidate','Used'], index=CandidatesList["IdCandidate"].values)
    boolList = [False] * (len(CandidatesList))
    CandidatesUsedList['Used'] = boolList
    CandidatesUsedList['IdCandidate']=CandidatesList['IdCandidate'].values
    return CandidatesUsedList



def getDataStructures(tree,df,window_size,k,verbose):

    #INPUT: Deicsion Tree, training dataset, sliding window size, # of motifs and discords extracted from each Ts

    #OUTPUT: Dataframe containing all the candidates extracted, the number of candidates extracted

    # transform target value into number
    le = LabelEncoder()
    num_classes = le.fit_transform(df['target'])
    df['target'] = num_classes


    diz = {'IdTs':[],'IdCandidate':[],'startingPosition':[],'M/D':[]}
    numberOfMotifs=0
    numberOfDiscords=0

    # Extract motifs and discords

    if(verbose):
        print('start computing MP, MPI')

    counter=0 #incremental counter for candidates
    for i in range(len(df)):
        if(tree.warningDetected==True and i % 100 == 0):
            if (verbose):
                print('computing Ts #: '+str(i))

        Ts = np.array(df.iloc[i][:-2].values) #-2 perche rimuovo l'attributo target e index inserito precedentemente

        tupla= retrieve_all(tree,Ts,window_size,k)
        mp, mot, motif_dist, dis = tupla



        for j in range(len(mot)):
            diz['IdTs'].insert(counter,df.iloc[i]['TsIndex'])
            diz['IdCandidate'].insert(counter, counter)
            diz['startingPosition'].insert(counter, mot[j][0])
            diz['M/D'].insert(counter, 0)
            numberOfMotifs +=1
            counter+=1

        for j in range(len(dis)):
            diz['IdTs'].insert(counter, df.iloc[i]['TsIndex'])
            diz['IdCandidate'].insert(counter, counter)
            diz['startingPosition'].insert(counter, dis[j])
            diz['M/D'].insert(counter, 1)
            numberOfDiscords +=1
            counter+=1


    CandidatesList = pd.DataFrame(diz)
    if (verbose == True):
        print('Candidati estratti: ')
        print(CandidatesList)
        print('numberOfMotif: %d, numberOfDiscord: %d \n'% (numberOfMotifs, numberOfDiscords))
        print('\n')

    return CandidatesList, numberOfMotifs, numberOfDiscords



def computeSubSeqDistance(tree, TsIndexList ,CandidatesList,window_size):

    #INPUT: Decision tree, list of Ts Index, list of candidates

    #OUTPUT: Training dataset such that: each row is a Ts, each column is a candidate, each cell is the Min euclidean dist btw the Ts and candidate

    columnsList = CandidatesList['IdCandidate'].values
    columnsList = columnsList.astype(int)
    dfForDTree = pd.DataFrame(columns=columnsList, index=range(0, len(TsIndexList)))
    dfForDTree['TsIndex'] = None
    dfForDTree['class'] = None


    #for each Ts, retireive each candidate and compute the min euclidean distance
    for i in range(len(TsIndexList)):
        # acquisisco la Ts di cui calcolare distanza
        TsIndexValue=TsIndexList[i]

        TsToCompare=np.array(tree.dfTrain[tree.dfTrain['TsIndex']==TsIndexValue].values)

        TsToCompare=TsToCompare[0]
        classValue = TsToCompare[len(TsToCompare)-2] #class value is always the penultimate value, Ts index index is last
        TsToCompare = TsToCompare[:len(TsToCompare)-2]

        dfForDTree['TsIndex'].iloc[i] = TsIndexValue
        dfForDTree['class'].iloc[i] = classValue


        for j in range (len(CandidatesList)):
            IdCandidate=CandidatesList.iloc[j]['IdCandidate']
            IdTsCandidate=CandidatesList.iloc[j]['IdTs']
            startingPosition=CandidatesList.iloc[j]['startingPosition']

            TsContainingCandidate = np.array(tree.dfTrain[tree.dfTrain['TsIndex'] == IdTsCandidate].values)
            TsContainingCandidate=TsContainingCandidate[0]
            TsContainingCandidate = TsContainingCandidate[:len(TsContainingCandidate) - 2]

            if (tree.warningDetected):
                Dp = distanceProfile.naiveDistanceProfile(TsContainingCandidate, int(startingPosition),
                                                          window_size, TsToCompare)
            else:
                Dp = distanceProfile.massDistanceProfile(TsContainingCandidate, int(startingPosition),
                                                         window_size, TsToCompare)
            minValueFromDProfile = min(Dp[0])  # Dp[0] contiene il Dp effettivo

            dfForDTree[int(IdCandidate)].iloc[i] = minValueFromDProfile



    return dfForDTree







#built differently because its called once on test dataset (DframeTest)
def computeSubSeqDistanceForTest(tree,datasetTest, CandidatesListTest):

    #INPUT: Decision tree, list of Ts Index belonging to the test set, list of candidates chosen by the Decision Tree

    #OUTPUT: Test dataset such that: each row is a Ts, each column is a candidate, each cell is the Min euclidean dist btw the Ts and candidate

    # quantifico il num di candidati usati dall'albero e in base a tale valore genero colonne per dfForDTree
    columnsList = CandidatesListTest['IdCandidate'].values
    columnsList = columnsList.astype(int)
    dfForDTreeTest = pd.DataFrame(columns=columnsList, index=range(0, len(datasetTest)))
    dfForDTreeTest['class'] = None


    # per ogni Ts, scandisco ogni candidato e calcolo la distanza minore
    for i in range(len(datasetTest)):
        # acquisisco la Ts
        TsToCompare = np.array(datasetTest.iloc[i].values)
        classValue = TsToCompare[len(TsToCompare) - 2]  # la classe è sempre il penultimo attributo
        TsToCompare = TsToCompare[:len(TsToCompare) - 2]  # la serie è ottenuta rimuovendo i due ultimi attributi
        #I VALORI (-1, -2) SONO DIVERSI DA QUELLI USATI IN COMPUTE NORMALE, PERCHE QUI NON PASSO LA STRUTTURA A GETDATASTRUCTURES => NON AGGIUNGO COLONNA TS INDEX

        dfForDTreeTest['class'].iloc[i] = classValue
        counter = 0 #scandisco candidate list (prima motif poi discord) incrementando counter -> cosi prenderò il candidato counter-esimo
        # scandisco e calcolo distanza dai candidati

        for z in range(len(CandidatesListTest)):
            IdCandidate = CandidatesListTest.iloc[z]['IdCandidate']
            IdTsCandidate = CandidatesListTest.iloc[z]['IdTs']
            startingPosition = CandidatesListTest.iloc[z]['startingPosition']

            TsContainingCandidate = np.array(tree.dfTrain[tree.dfTrain['TsIndex'] == IdTsCandidate].values)
            TsContainingCandidate = TsContainingCandidate[0]
            TsContainingCandidate = TsContainingCandidate[:len(TsContainingCandidate) - 2]

            if (tree.warningDetected):
                Dp = distanceProfile.naiveDistanceProfile(TsContainingCandidate, int(startingPosition),
                                                          tree.window_size, TsToCompare)
            else:
                Dp = distanceProfile.massDistanceProfile(TsContainingCandidate, int(startingPosition),
                                                         tree.window_size, TsToCompare)



            minValueFromDProfile = min(Dp[0])  # Dp[0] contiene il Dp effettivo
            dfForDTreeTest[int(IdCandidate)].iloc[i] = minValueFromDProfile

    le = LabelEncoder()
    num_classes = le.fit_transform(dfForDTreeTest['class'])
    dfForDTreeTest['class'] = num_classes

    return dfForDTreeTest



#Apply the K-Medoids to the candidatesList
def reduceNumberCandidates(tree,CandidatesList,returnOnlyIndex):

    #INPUT: Decision Tree, list of candidates, options

    #OUTPUT: List of the candidates chosen as medoids

    #returnOnlyIndex = True => returns only candidates index
    #                  False => returns candidates

    if(tree.n_clusters>= len(CandidatesList) or len(CandidatesList)==0):
        if(tree.verbose):
            print('Nessun clustering necessario su CandidatesList')
            print('len CandidatesList: %s num cluster: %s \n' % (len(CandidatesList),tree.n_clusters))
        if(returnOnlyIndex):
            return np.arange(0,len(CandidatesList))
        else:
            return CandidatesList


    verboseretireveCandidatesSubSeq=False

    columnsList = list(['IdTs', 'IdCandidate', 'startingPosition', 'M/D'])
    prefix = 'att'
    for i in range(tree.window_size):
        columnsList.append(prefix + str(i))
    CandidatesSubSeq = pd.DataFrame(columns=columnsList, index=range(len(CandidatesList)))

    counter = 0
    for j in range(len(CandidatesList)):  # CANDIDATI
        startingIndex = CandidatesList.iloc[j]['startingPosition']  # indice di inizio del motif
        indexTsContainingCandidateShapelet = CandidatesList.iloc[j]['IdTs']

        TsContainingCandidateShapelet = tree.dfTrain[tree.dfTrain['TsIndex'] == indexTsContainingCandidateShapelet]
        TsContainingCandidateShapelet = TsContainingCandidateShapelet.values
        TsContainingCandidateShapelet = TsContainingCandidateShapelet[0][:-2]

        subSeqCandidate = TsContainingCandidateShapelet[startingIndex:startingIndex + tree.window_size]



        CandidatesSubSeq.iloc[counter]['IdTs'] = CandidatesList.iloc[j]['IdTs']
        CandidatesSubSeq.iloc[counter]['IdCandidate'] = CandidatesList.iloc[j]['IdCandidate']
        CandidatesSubSeq.iloc[counter]['startingPosition'] = startingIndex
        CandidatesSubSeq.iloc[counter]['M/D'] = CandidatesList.iloc[j]['M/D']
        for z in range(tree.window_size):
            CandidatesSubSeq.iloc[counter]['att' + str(z)] = subSeqCandidate[z]
        counter += 1




    if(verboseretireveCandidatesSubSeq):
        print('subseq dei candidati estratti')
        #subseq dei candidati estratti (rispettivamente motifs e poi discords)
        print(CandidatesSubSeq)


    CandidateMedoids=[]


    # indici all interno di candDfMotifs & candDfDiscords dei candidati scelti come medoidi
    CandidateMedoids = runKMedoids(tree,CandidatesSubSeq, tree.n_clusters)
    if(verboseretireveCandidatesSubSeq):
        print('indici all interno di CandidatesList scelti come medoidi ')
        print(CandidateMedoids)



    CandidatesSubSeq=CandidatesSubSeq.iloc[CandidateMedoids]
    CandidatesSubSeq=CandidatesSubSeq[['IdTs', 'IdCandidate', 'startingPosition', 'M/D']]
    CandidatesSubSeq=CandidatesSubSeq.reset_index(drop=True)


    if(returnOnlyIndex):
        return CandidateMedoids
    else:
        return CandidatesSubSeq







def retrieve_all2(Ts,window_size,k):  # fornita la Ts calcola e restituisce mp, motifs, motifs_distances e discords
    Ts = Ts[:len(Ts)-1]  # rimuovo l'attributo "classe"

    dfMP = pd.DataFrame(Ts).astype(float)  # genero Dframe per lavorarci su, DA CAPIRE PERCHE SERVE FLOAT
    mp, mpi = matrixProfile.stomp(dfMP[0].values, window_size)  # OK STOMP

    # PREPARO TUPLA DA PASSARE ALLA FUN MOTIF (RICHIEDE TUPLA FATTA DA MP E MPI)
    tupla = mp, mpi

    mot, motif_dist = motifs.motifs(dfMP[0].values, tupla, k)

    # CALCOLO MOTIFS
    print('Motifs starting position: ' + str(mot) + ' Motifs values (min distances): ' + str(motif_dist))

    # CALCOLO DISCORDS
    dis = discords(mp, window_size, k)
    print('Discords starting position: ' + str(dis))

    tupla = mp, mot, motif_dist, dis
    return tupla


# riceve la lista di coppie dei motifs per ogni record(Ts), e resittuisce lista di valori singoli

def candidateFilter2(CandidateList):
    l2 = np.array([])
    for i in range(len(CandidateList['Motif'])):  # per ogni entry (per ogni record)
        numMotif = len(CandidateList['Motif'].iloc[i])
        # print(numMotif)
        for j in range(numMotif):  # per ogni lista di motif
            l1 = CandidateList['Motif'].iloc[i]  # prima lista
            l2 = np.append(l2, l1[j][0])  # prendo primo valore di ogni lista

        CandidateList['Motif'].iloc[i] = l2
        l2 = np.array([])  # svuoto array

    return CandidateList