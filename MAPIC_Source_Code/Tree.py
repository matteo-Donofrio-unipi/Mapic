import pandas as pd
import numpy as np
from matrixprofile import *
from matrixprofile.discords import discords
from matplotlib import pyplot as plt
from binarytree import Node
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from Tools import  computeSubSeqDistance,reduceNumberCandidates
from PlotLibrary import plotDataAndShapelet


class Tree:
    def __init__(self,candidatesGroup,maxDepth,minSamplesLeaf,removeUsedCandidate,window_size,k,useClustering,n_clusters,warningDetected, verbose ):
        self.candidatesGroup=candidatesGroup
        self.maxDepth=maxDepth
        self.minSamplesLeaf=minSamplesLeaf
        self.removeUsedCandidate=removeUsedCandidate
        self.window_size=window_size
        self.k=k
        self.useClustering=useClustering
        self.n_clusters=n_clusters
        self.OriginalCandidatesUsedListTrain=[]
        self.warningDetected=warningDetected
        self.verbose=verbose
        self.attributeList=list()
        self.SseList=list()
        self.IterationList = list()


    def computeEntropy(self,dataset):

        #INPUT: Dataset contining Ts and their label

        #OUTPUT: Entropy of the Dataset

        value, counts = np.unique(dataset['class'], return_counts=True)
        actualEntropy = entropy(counts, base=2)
        return actualEntropy


    def computeGain(self,entropyParent, LenDatasetParent, Dleft, Dright):

        #INPUT: Entropy of the parent node, lenght parent node, left and right child node

        #OUTPUT: Gain entropy btw parent entropy and the weighed summation of the child entropy

        entropyLeft = self.computeEntropy(Dleft)
        entropyRight = self.computeEntropy(Dright)
        gain = entropyParent
        summation = (
                    ((len(Dleft) / LenDatasetParent) * entropyLeft) + ((len(Dright) / LenDatasetParent) * entropyRight))
        gain = gain - summation
        return gain





    def split(self,dataset, attribute, value):

        #INPUT: dataset,attribute and value on which split the dataset

        #OUTPUT: left and right dataset obtained from the split

        columnsList = dataset.columns.values
        attribute=str(attribute)
        #value=str(value)

        dizLeft=dataset[dataset[int(attribute)] < value]
        dizRight = dataset[dataset[int(attribute)] >= value]

        dizLeft = dizLeft.reset_index(drop=True)
        dizRight = dizRight.reset_index(drop=True)


        return dizLeft, dizRight




    # riceve dframe con mutual_information(gain) e in base al candidatesGroup scelto, determina il miglior attributo su cui splittare
    # che non è stato ancora utilizzato
    def getBestIndexAttribute(self,CandidatesUsedListTrain,vecMutualInfo,verbose):
        # ordino i candidati in base a gain decrescente

        vecMutualInfo = vecMutualInfo.sort_values(by='gain', ascending=False)

        # scandisco i candidati fino a trovare il candidato con miglior gain che non è ancora stato usato

        bestIndexAttribute = -1
        i = 0

        # cicla fin quando trova candidato libero con gain maggiore
        while (bestIndexAttribute == -1 and i < len(vecMutualInfo)):
            attributeToVerify = int(vecMutualInfo.iloc[i]['IdCandidate'])
            if (CandidatesUsedListTrain.loc[attributeToVerify]['Used']==False):
                bestIndexAttribute = attributeToVerify
                splitValue = vecMutualInfo.iloc[i]['splitValue']

                CandidatesUsedListTrain.loc[CandidatesUsedListTrain["IdCandidate"]==attributeToVerify,"Used"] = True  # settando a true il candidato scelto, non sarà usato in seguito
                if (verbose):
                    print('gain: ' + str(vecMutualInfo.iloc[i]['gain']))
            else:
                i += 1

        return bestIndexAttribute, splitValue




    #Given the dataset, compute for each candidate: the best treshold value on which split and the relative difference entropy gained
    def computeMutualInfo(self,datasetForMutual,verbose):
        # cerca attributo, il cui relativo best split value massimizza l'information gain nello split

        #INPUT: Dataset


        #OUTPUT: Dataframe such that each row contains the best treshold and the relative difference entropy gained, on a specific candidate

        columns = datasetForMutual.columns
        dframe = pd.DataFrame(columns=['IdCandidate', 'splitValue', 'gain'],
                              index=range(len(columns) - 2))  # -1 cosi non prendo attr=class e TsIndex
        entropyParent = self.computeEntropy(datasetForMutual)

        # per ogni attributo, ordino il dframe sul suo valore
        # scandisco poi la y e appena cambia il valore di class effettuo uno split, memorizzando il best gain

        for i in range(len(columns) - 2):  # scandisco tutti gli attributi tranne 'class'
            bestGain = -1
            bestvalueForSplit = 0
            previousClass = -1  # deve essere settato ad un valore non presente nei class value
            attribute = columns[i]
            if (verbose):
                print('COMPUTE attr: ' + str(attribute))
            datasetForMutual = datasetForMutual.sort_values(by=attribute, ascending=True)

            y = datasetForMutual['class']

            for j in range(len(y)):
                if (j == 0):
                    previousClass = y[j]
                    continue
                else:
                    if (y[j] != previousClass):
                        testValue = datasetForMutual.iloc[j][attribute]
                        Dleft, Dright = self.split(datasetForMutual, attribute, testValue)
                        actualGain = self.computeGain(entropyParent, len(datasetForMutual), Dleft, Dright)
                        if (actualGain > bestGain):
                            bestGain = actualGain
                            bestvalueForSplit = testValue

                    previousClass = y[j]
                    # memorizzo in posizione i-esima lo split migliore e relativo gain

            dframe.iloc[i]['splitValue'] = bestvalueForSplit
            dframe.iloc[i]['gain'] = bestGain

        dframe['IdCandidate'] = columns[:-2]

        return dframe



    #Find the Shapelet and relative OptimalSplitPoint (treshold value) on which split the dataset
    def findBestSplit(self,dfForDTree, verbose=False):

        #INPUT: Training Dataset

        #OUTPUT: CandidateIndex, value, left and right child dataset


        vecMutualInfo = self.computeMutualInfo(dfForDTree,verbose)
        if (verbose == True):
            print('vec mutual info calcolato: ')
            print(vecMutualInfo)

        # based on the "removeUsedCandidate" parameter value, its chosen the best candidate on which split
        if (self.removeUsedCandidate == 1):
            indexBestAttribute, bestValueForSplit  = self.getBestIndexAttribute(self.OriginalCandidatesUsedListTrain,vecMutualInfo,verbose)
        else:  # se non rimuovo candidati, mi basta prendere il primo
            vecMutualInfo = vecMutualInfo.sort_values(by='gain', ascending=False)
            indexBestAttribute = vecMutualInfo.iloc[0]['IdCandidate']
            bestValueForSplit = vecMutualInfo.iloc[0]['splitValue']
            if (verbose):
                print('gain: ' + str(vecMutualInfo.iloc[0]['gain']))

        if (verbose == True):
            print('BEST attribute | value')
            print(indexBestAttribute, bestValueForSplit)

        splitValue = bestValueForSplit
        Dleft, Dright = self.split(dfForDTree,str(indexBestAttribute), bestValueForSplit)

        return [indexBestAttribute, splitValue, Dleft, Dright]

    # SPLIT MASTER
    # funzione ricorsiva che implementa la creazione dell'albero di classificazione
    # memorizza in ogni nodo: attributo, valore attributo su cui splitto, entropia nodo, num pattern
    # memorizza in ogni foglia: entropia nodo, num pattern, classe nodo

    #Recursive function that builds the Decision Tree
    def buildTree(self,actualNode, dataset, depth,verbose):

        #Base case
        boolValue = self.checkIfIsLeaf(dataset)
        if (len(dataset) < self.minSamplesLeaf or depth >= self.maxDepth or boolValue == True):
            average = sum(dataset['class'].values) / len(dataset['class'].values)
            classValue = round(average)
            numPattern = len(dataset)
            entropy = self.computeEntropy(dataset)

            nodeInfo = list()
            nodeInfo.append(classValue)
            nodeInfo.append(numPattern)
            nodeInfo.append(entropy)

            actualNode.data = nodeInfo
            actualNode.value = classValue
            actualNode.left = None
            actualNode.right = None
            return

        else: #Recursive case

            returnList = self.findBestSplit(dataset,verbose)
            indexChosenAttribute = returnList[0]
            attributeValue = returnList[1]
            Dleft = returnList[2]
            Dright = returnList[3]
            numPattern = len(dataset)
            entropy = self.computeEntropy(dataset)
            self.attributeList.append(indexChosenAttribute)

            #Store in each node values about split

            nodeInfo = list()
            nodeInfo.append(attributeValue)
            nodeInfo.append(numPattern)
            nodeInfo.append(entropy)
            actualNode.data = nodeInfo
            actualNode.value = (int(indexChosenAttribute))

            if (verbose):
                print('DLEFT & DRIGHT INIZIALMENTE')
                print(Dleft)
                print(Dright)

            if(self.useClustering):




                # APPLY K-MEDOIDS FOR DLEFT------------------------------------------------------

                TsIndexLeft = Dleft['TsIndex']  # TsIndex contenute in Dleft

                #Retrieve medoids among all candidates

                CandidatesListLeft = self.OriginalCandidatesListTrain['IdTs'].isin(
                    TsIndexLeft)  # setta a True gli indici dei candidati che sono stati generati dalle Ts contenute in Dleft

                CandidateToCluster = self.OriginalCandidatesListTrain[
                    CandidatesListLeft]  # estraggo i candidati da OriginalCandidatesListTrain, che sono generati dalle Ts in Dleft
                CandidateToCluster = CandidateToCluster.reset_index(drop=True)

                indexChoosenMedoids = reduceNumberCandidates(self, CandidateToCluster,
                                                             returnOnlyIndex=True)  # indici di OriginalCandidatesListTrain conteneti candidati da mantenere

                CandidateToCluster = CandidateToCluster.iloc[indexChoosenMedoids] #candidati da mantenere

                if (verbose):
                    print('CANDIDATI RIMASTI IN BUILD')
                    print(CandidateToCluster)

                # Compute distances btw Ts and chosen medoids
                Dleft = computeSubSeqDistance(self, TsIndexLeft, CandidateToCluster, self.window_size)


                # APPLY K-MEDOIDS FOR DRIGHT------------------------------------------------------

                TsIndexRight = Dright['TsIndex']  # TsIndex contenute in Dleft

                # Retrieve medoids among all candidates

                CandidatesListRight = self.OriginalCandidatesListTrain['IdTs'].isin(
                    TsIndexRight)  # # setta a True gli indici dei candidati che sono stati generati dalle Ts contenute in Dright

                CandidateToCluster = self.OriginalCandidatesListTrain[
                    CandidatesListRight]   # estraggo i candidati da OriginalCandidatesListTrain, che sono generati dalle Ts in Dright
                CandidateToCluster = CandidateToCluster.reset_index(drop=True)

                indexChoosenMedoids = reduceNumberCandidates(self, CandidateToCluster,
                                                             returnOnlyIndex=True)  # indici di OriginalCandidatesListTrain conteneti candidati da mantenere

                CandidateToCluster = CandidateToCluster.iloc[indexChoosenMedoids] #candidati da mantenere

                if (verbose):
                    print('CANDIDATI RIMASTI IN BUILD')
                    print(CandidateToCluster)

                # Compute distances btw Ts and chosen medoids
                Dright = computeSubSeqDistance(self, TsIndexRight, CandidateToCluster, self.window_size)

                if (verbose):
                    print('DLEFT & DRIGHT DOPO IL CLUSTERING')
                    print(Dleft)
                    print(Dright)



            #Recursive call
            if (len(Dleft) > 0):
                actualNode.left = Node(int(indexChosenAttribute))
                self.buildTree(actualNode.left, Dleft, depth + 1, verbose)

            if (len(Dright) > 0):
                actualNode.right = Node(int(indexChosenAttribute))
                self.buildTree(actualNode.right, Dright, depth + 1, verbose)




    def checkIfIsLeaf(self,dataset):
        #INPUT: Dataset

        #OUTPUT: True if the node containing this dataset can be defined as a Leaf

        isLeaf = True
        entropy = self.computeEntropy(dataset)
        if (entropy > 0):
            isLeaf = False
        return isLeaf






    # Start to define and train the model
    def fit(self,dfForDTree,verbose):

        #INPUT: Training dataset

        #OUTPUT: Trained Decision Tree

        # inizio algo per nodo radice
        returnList = self.findBestSplit(dfForDTree,False)
        indexChosenAttribute = returnList[0]
        attributeValue = returnList[1]
        Dleft = returnList[2]
        Dright = returnList[3]
        self.attributeList.append(int(indexChosenAttribute))
        root = Node(int(indexChosenAttribute))
        numPattern = len(dfForDTree)
        entropy = self.computeEntropy(dfForDTree)

        # memorizzo nel nodo l'attributo, il valore e altre info ottenute dallo split

        nodeInfo = list()
        nodeInfo.append(attributeValue)
        nodeInfo.append(numPattern)
        nodeInfo.append(entropy)
        root.data = nodeInfo

        root.left = Node(int(indexChosenAttribute))
        root.right = Node(int(indexChosenAttribute))

        if (verbose):
            print('DLEFT & DRIGHT INIZIALMENTE')
            print(Dleft)
            print(Dright)

        if(self.useClustering):

            # APPLY K-MEDOIDS FOR DLEFT------------------------------------------------------


            TsIndexLeft = Dleft['TsIndex']  # TsIndex contenute in Dleft


            # Retrieve medoids among all candidates
            CandidatesListLeft = self.OriginalCandidatesListTrain['IdTs'].isin(
                TsIndexLeft)  #  setta a True gli indici dei candidati che sono stati generati dalle Ts contenute in Dleft

            CandidateToCluster = self.OriginalCandidatesListTrain[
                CandidatesListLeft]  # estraggo i candidati da OriginalCandidatesListTrain, che sono generati dalle Ts in Dleft

            CandidateToCluster = CandidateToCluster.reset_index(drop=True)

            indexChoosenMedoids = reduceNumberCandidates(self, CandidateToCluster,
                                                         returnOnlyIndex=True)  # indici di OriginalCandidatesListTrain conteneti candidati da mantenere

            CandidateToCluster = CandidateToCluster.iloc[indexChoosenMedoids]

            if (verbose):
                print('CANDIDATI RIMASTI IN FIT')
                print(CandidateToCluster)

            # Compute distances btw Ts and chosen medoids
            Dleft = computeSubSeqDistance(self, TsIndexLeft, CandidateToCluster, self.window_size)



            # APPLY K-MEDOIDS FOR DRIGHT------------------------------------------------------

            TsIndexRight = Dright['TsIndex']  # TsIndex contenute in Dleft

            # Retrieve medoids among all candidates
            CandidatesListRight = self.OriginalCandidatesListTrain['IdTs'].isin(
                TsIndexRight) #  setta a True gli indici dei candidati che sono stati generati dalle Ts contenute in Dright

            CandidateToCluster = self.OriginalCandidatesListTrain[
                CandidatesListRight]  # estraggo i candidati da OriginalCandidatesListTrain, che sono generati dalle Ts in Dleft
            CandidateToCluster = CandidateToCluster.reset_index(drop=True)

            indexChoosenMedoids = reduceNumberCandidates(self,CandidateToCluster,returnOnlyIndex=True)  # indici di OriginalCandidatesListTrain conteneti candidati da mantenere

            CandidateToCluster = CandidateToCluster.iloc[indexChoosenMedoids]

            if (verbose):
                print('CANDIDATI RIMASTI IN FIT')
                print(CandidateToCluster)

            # Compute distances btw Ts and chosen medoids
            Dright = computeSubSeqDistance(self, TsIndexRight, CandidateToCluster, self.window_size)

            if (verbose):
                print('DLEFT & DRIGHT DOPO IL CLUSTERING')
                print(Dleft)
                print(Dright)


        # Recursive call
        if (len(Dleft) > 0):
            self.buildTree(root.left, Dleft, 1, verbose)
        if (len(Dright) > 0):
            self.buildTree(root.right, Dright, 1, verbose)
        Tree.Root = root





    def printAll(self,Root):
        if (Root.left == None and Root.right == None):
            print('leaf')
        print('Node: ' + str(Root.value))
        df = Root.data
        print(df)
        print("\n")
        if (Root.left != None):
            self.printAll(Root.left)
        if (Root.right != None):
            self.printAll(Root.right)


    def predict(self,testDataset, root, printClassifiedInstances):

        #INPUT: Test Dataset, Decision tree root, option

        #OUTPUT: List of test label, list of predicted label

        numAttributes = len(testDataset.columns.values)
        numAttributes -= 1  # per prendere solo gli attributi utili a xTest
        yTest = testDataset.iloc[:]['class'].values
        yPredicted = np.zeros(len(yTest))
        xTest = testDataset.iloc[:, np.r_[:numAttributes]]

        #Dictionary in which store shapelet for plot the classified instances
        self.ShapeletDf=pd.DataFrame(columns=['IdShapelet','distance','majorMinor','startingIndex'],index=range(0,len(self.dTreeAttributes)))

        self.Shapelet = pd.DataFrame(columns=['IdShapelet','Shapelet'],index=range(0,len(self.dTreeAttributes)))



        # make prediction for each pattern
        for i in range(len(xTest)):
            pattern = xTest.iloc[i]
            if(printClassifiedInstances==True):
                yPredicted[i] = self.treeExplorerPrint(pattern, root,0,i)
                plotDataAndShapelet(self,i,yPredicted[i])
                printClassifiedInstances = False #set to False for plot only 1 instances
            else:
                yPredicted[i] = self.treeExplorer(pattern, root) #predict without plot


        yTest = yTest.astype(int)
        yPredicted = yPredicted.astype(int)

        return yTest, yPredicted



    #Recursive function, given a pattern, classifies it by exploring the Decision tree
    def treeExplorer(self,pattern, node):
        # caso base, node è foglia
        if (node.left == None and node.right == None):
            return int(node.data[0])
        else:
            # caso ricorsivo
            attr = int(node.value)
            if (pattern[attr] < node.data[0]):
                return self.treeExplorer(pattern, node.left)
            else:
                return self.treeExplorer(pattern, node.right)

    # Recursive function, given a pattern, classifies it by exploring the Decision tree
    # store the information needed for the plot
    def treeExplorerPrint(self, pattern, node,counter,i):
        # Base case
        if (node.left == None and node.right == None):
            return int(node.data[0])
        else:
            # recursive case
            self.counter=counter+1 #+1 perche parte da 0 e voglio il numero effettivo
            attr = int(node.value)



            idTsShapelet = self.dTreeAttributes[self.dTreeAttributes['IdCandidate'] == int(attr)]["IdTs"].values
            idTsShapelet = idTsShapelet[0]

            startingPosition = self.dTreeAttributes[self.dTreeAttributes['IdCandidate'] == int(attr)][
                    "startingPosition"].values
            startingPosition = startingPosition[0]

            TsContainingShapelet = np.array(self.dfTrain[self.dfTrain['TsIndex'] == idTsShapelet].values)
            TsContainingShapelet = TsContainingShapelet[0]
            TsContainingShapelet = TsContainingShapelet[:len(TsContainingShapelet) - 2]

            if (self.warningDetected):
                Dp = distanceProfile.naiveDistanceProfile(TsContainingShapelet, int(startingPosition),
                                                              self.window_size, self.TsTestForPrint[i])
            else:
                Dp = distanceProfile.massDistanceProfile(TsContainingShapelet, int(startingPosition),
                                                             self.window_size, self.TsTestForPrint[i])

            val, idx = min((val, idx) for (idx, val) in enumerate(Dp[0]))

            self.ShapeletDf.iloc[counter]['IdShapelet']=attr
            self.ShapeletDf.iloc[counter]['distance'] = val
            self.ShapeletDf.iloc[counter]['startingIndex'] = idx

            self.Shapelet.iloc[counter]['IdShapelet']=attr
            self.Shapelet.iloc[counter]['Shapelet']=TsContainingShapelet[startingPosition:startingPosition+self.window_size]


            if (pattern[attr] < node.data[0]):
                self.ShapeletDf.iloc[counter]['majorMinor'] = -1
                counter += 1
                return self.treeExplorerPrint(pattern, node.left,counter,i)
            else:
                self.ShapeletDf.iloc[counter]['majorMinor'] = 1
                counter += 1
                return self.treeExplorerPrint(pattern, node.right,counter,i)





