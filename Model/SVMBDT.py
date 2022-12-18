"""
SVMBDT.py: 

This program contains the the functions used to compute Support Vector Machine
with Binary Desion Tree (SVMBDT). The main functions are SvmDesionTreeTrain() and
SvmDesionTreePredict() that are designed to train and predict from data of 
hyperspectral images. 

"""
__author__ = "Jonas Gjendem Røysland"
__email__ = "jonasroy@stud.ntnu.no"
__date__ = "2022/12/18"
__version__ = "1.0.0"

from HelperFunctions_ import classesInLabels
import numpy as np
import copy
import time

def SeperateDataLabels(data,labels, seperation): 

    """
    - Takes in the data and labels and divides into two different into the induvial tree_branch. 
    - Sepreation is in form [[A],[B]] = [[A_1, ..., A_n],[B_1, ..., B_n]] where it makes the subpicture with only two classes. 
    - The two classes is min(A) and min(B). 
    - Returns train_data and the corresponding train_labels. 
    """

    train_data = copy.deepcopy(data)
    train_labels = copy.deepcopy(labels)

    if(len(seperation) == 2): 

        classesInSeperation = []

        for i in range(len(seperation)): 
            for j in range(len(seperation[i])): 
                if(not(seperation[i][j] in classesInSeperation)): 
                    classesInSeperation.append(seperation[i][j])

        classesInSeperation.sort()
        classesInSeperation = np.array(classesInSeperation)
        cIL = np.array(classesInLabels(labels))

        if(not(np.array_equal(cIL, classesInSeperation))): 

            for i in range(len(cIL)): 
                if(not(cIL[i] in classesInSeperation)):
                    train_data = train_data[train_labels != cIL[i]]
                    train_labels = train_labels[train_labels != cIL[i]]

        if(len(classesInLabels(labels)) > 2):
            for i in range(len(seperation)): 
                for j in range(len(seperation[i])):
                    if(seperation[i][j] != min(seperation[i])):
                        train_labels[train_labels == seperation[i][j]] = min(seperation[i])
        
        return train_data, train_labels
        
    else: 
        print("The seperation needs to be 2 in length. Format: [[A],[B]] = [[A_1, ..., A_n],[B_1, ..., B_m]]")
        return -1



def BranchDataLabels(data, labels, tree_branches): 

    """
    - Takes in the train_data and train_labels and uses the tree_branches arcitecthure. 
    - tree_branches structure is a dictornary form of:
        - {0 : [[A_0,...,A_n],[B_0,...,B_n]], 1 : [[[A_0],[A_1,...,A_n]],[[B_0],[B_1,...,B_m]]], ... , N : {[[[A_(n-1)],[A_n]],[[B_(m-1)],[B_m]]]}
        - If the tree_branch structure is unsymmetrical of clases or branches implement the tree layer branch: [[],[[B_(m-1)],[B_m]]
    - Uses SeperateDataLabels() to seperate the classes in the different branches in the tree structure. 
    - Returns data_and_labels_branches that contains tree branch with seperated data and labels used for training for the SvmBranchModelTrain() function. 
    """

    if(len(tree_branches) > 0): 
        if(len(tree_branches[0]) == 2): 

            #Seperating the classes on the first branch of the data and the labels.
            data_and_labels_branches = {0 : SeperateDataLabels(data, labels, tree_branches[0])}
            
            for i in range(1,len(tree_branches)): 
                data_and_labels = []
                for j in range(len(tree_branches[i])): 
                    if(len(tree_branches[i][j]) == 2):
                            data_and_labels.append(SeperateDataLabels(data,labels,tree_branches[i][j]))
                    else: 
                        data_and_labels.append([])
                data_and_labels_branches[i] = data_and_labels

            return data_and_labels_branches
        
        else: 
            return -1

def SvmBranchModelTrain(data_and_labels_branches, svm_branch_models): 

    """
    Takes in the data_and_labels_branches and svm_branch_models: 
        - trains the indivuals svm models with branches of data_and_labels_branches
        - Is design for Sklearns functions SVC() and LinearSVC().
        - the svm_branch_models strucutre is a dictonary form of: 
            - {0 : SVC(), 1 : [SVC(), SVC()], ..., N : [SVC(),SVC()]}
            - If unsymmetric tree branch: [False, SVC()] 
    Returns svm_branch_models_trained
    """

    svm_branch_models_trained = {}

    svm = svm_branch_models[0]
    train_data, train_labels = data_and_labels_branches[0]
    svm.fit(train_data, train_labels)
    svm_branch_models_trained[0] = svm

    for i in range(1,len(data_and_labels_branches)): 
        svm_models = []
        for j in range(len(data_and_labels_branches[i])): 
            if(len(data_and_labels_branches[i][j]) == 2): 
                train_data, train_labels = data_and_labels_branches[i][j]
                svm = copy.deepcopy(svm_branch_models[i][j]) 
                svm.fit(train_data, train_labels)
                svm_models.append(svm)
            else: 
                svm_models.append(False)
        svm_branch_models_trained[i] = svm_models

    return svm_branch_models_trained
    

def SvmBranchModelPredict(data, svm_branch_models, tree_branch, sub_data_return = 0): 

    """
    Takes in data, svm_branch_models, tree_branch. 
        - Predict data picture and labels the data into two classes. 
        - Seperate the data and labels to two different branches. 
        - Predicts the two sub data and places it to each other sub branch. 
            - Does this after the tree_branch structure. 
    
    Return the predicted_branch_labels, which is tree branch whit different sub labels
        - For returning also sub_data_branch, sub_data_return = 1 
    """

    test_data = copy.deepcopy(data)

    predicted_branch_labels = {}
    sub_data_branch = {}
    sub_data_sub_branch = []

    svm = svm_branch_models[0]
    start_time = time.time()
    yout = svm.predict(test_data)
    stop_time = time.time()

    print("The first branch: " + str(round(stop_time - start_time,3)))

    predicted_branch_labels[0] = yout
    sub_data_branch[0] = test_data

    sub_data_1 = test_data[yout == min(svm.classes_)]
    sub_data_2 = test_data[yout == max(svm.classes_)]

    branch_classes_1 = [-1] 
    branch_classes_2 = [-1] 

    sub_data_sub_branch = [[],[]]

    if((len(tree_branch[1][0]) > 0)): 
        branch_classes_1 = tree_branch[1][0][0] + tree_branch[1][0][1]
    
    if((len(tree_branch[1][1]) > 0)): 
        branch_classes_2 = tree_branch[1][1][0] + tree_branch[1][1][1]

    if(min(yout) in branch_classes_1): 
        sub_data_sub_branch[0] = sub_data_1
    
    if(max(yout) in branch_classes_1): 
        sub_data_sub_branch[0] = sub_data_2

    if(min(yout) in branch_classes_2): 
        sub_data_sub_branch[1] = sub_data_1
    
    if(max(yout) in branch_classes_2): 
        sub_data_sub_branch[1] = sub_data_2

    sub_data_branch[1] = sub_data_sub_branch

    for i in range(1,len(svm_branch_models)):
        predicted_labels = [[],[]]
        sub_data_sub_branch = [[],[]]
        for j in range(len(svm_branch_models[i])): 
            if(not(svm_branch_models[i][j] == False)):           
                sub_data = sub_data_branch[i][j]
                svm = svm_branch_models[i][j]
                
                start_time = time.time()
                yout = svm.predict(sub_data)
                stop_time = time.time()

                predicted_labels[j] = yout 

                if(i < len(svm_branch_models)-1): 
                    branch_classes_1 = [-1]
                    #branch_classes_2 = [-1]

                    sub_data_1 = sub_data[yout == min(yout)]
                    sub_data_2 = sub_data[yout == max(yout)]

                    if((len(tree_branch[i+1][j]) > 0)): 
                        branch_classes_1 = tree_branch[i+1][j][0] + tree_branch[i+1][j][1]
    
                    if(min(yout) in branch_classes_1): 
                        sub_data_sub_branch[j] = sub_data_1
    
                    if(max(yout) in branch_classes_1): 
                        sub_data_sub_branch[j] = sub_data_2
                    
                    sub_data_branch[i+1] = sub_data_sub_branch


        predicted_branch_labels[i] = predicted_labels


    if(sub_data_return == 0): 
        return predicted_branch_labels
    
    if(sub_data_return == 1): 
        return predicted_branch_labels, sub_data_branch



def CombineLabels(predicted_branch_labels, tree_branch): 

    """
    Takes in the predicted_branch_labels and places the labels to the correct placing on the data picture. 
     - Checks if the are difference between the sub labels between the two branches. 
     Returns the predicted label with all the classes. 
    """

    pl = copy.deepcopy(predicted_branch_labels)

    for i in range(len(pl)-1,1,-1):
        for j in range(len(pl[i])):
            if(len(pl[i-1][j]) > 0): 
                if(len(pl[i][j]) > 0): 
                    if(min(pl[i][j]) in pl[i-1][j]): 
                        yout_branch = pl[i-1][j]
                        yout_sub_branch = pl[i][j]
                        class_lenght_diff = int(np.abs(len(yout_branch[yout_branch == min(yout_sub_branch)]) - len(yout_sub_branch)))

                        #print("class length diff: " + str(class_lenght_diff))
                        
                        min_sub_value = min(yout_sub_branch)

                        for x in range(len(yout_branch)): 
                            if(class_lenght_diff < len(yout_branch)):
                                if(class_lenght_diff < len(yout_sub_branch)):  
                                    if(yout_branch[x] == min_sub_value): 
                                        yout_branch[x] = yout_sub_branch[class_lenght_diff]
                                        class_lenght_diff += 1
                        pl[i][j] = yout_branch
    #"""
    
    #Collecting the labels from the second branch layer to the first layer. 
    #Cobime the two second layers to the final labeling. 

    if(len(pl) > 1): 
        if(len(pl[1][0]) and not(len(pl[1][1]))): 
            yout_sub_1 = pl[1][0]
            yout_sub_2 = []
        
        elif(len(pl[1][1]) and not(len(pl[1][0]))):
            yout_sub_1 = []
            yout_sub_2 = pl[1][1]

        else: 
            yout_sub_1 = pl[1][0]
            yout_sub_2 = pl[1][1]

        yout = pl[0]

        if(yout_sub_1 == []): 
            class_diff = np.abs(len(yout[yout == min(tree_branch[0][1])]) - len(yout_sub_2))
            for x in range(len(yout)): 
                if(yout[x] in tree_branch[0][1]):
                    if(class_diff < len(yout_sub_2)):
                        yout[x] = yout_sub_2[class_diff] 
                        class_diff += 1
            #print(class_diff)

        elif(yout_sub_2 == []): 
            class_diff = np.abs(len(yout[yout == min(tree_branch[0][0])]) - len(yout_sub_1))
            for x in range(len(yout)): 
                if(yout[x] in tree_branch[0][0]):
                    if(class_diff < len(yout_sub_1)):
                        yout[x] = yout_sub_1[class_diff] 
                        class_diff += 1
            #print(class_diff)

        else:
            count_sub_1 = 0 
            count_sub_2 = 0 
            for x in range(len(yout)): 
                if(yout[x] in tree_branch[0][0]):
                    if(count_sub_1 < len(yout_sub_1)):
                        yout[x] = yout_sub_1[count_sub_1] 
                        count_sub_1 += 1
                if(yout[x] in tree_branch[0][1]):
                    if(count_sub_2 < len(yout_sub_2)): 
                        yout[x] = yout_sub_2[count_sub_2] 
                        count_sub_2 += 1
        return yout

    else: 
        return pl[0]



def RetrieveSubData(data,labels, sub_tree_branch): 

    """
    Gets the classes that are label in a training data. 
    Collects the corresponding pixels in the data. 
    Return train_data, train_labels that are sub-data for a sub-tree. 
    
    """


    train_data = copy.deepcopy(data)
    train_labels = copy.deepcopy(labels)

    seperation = np.array(sub_tree_branch[0] + sub_tree_branch[1])

    classes = classesInLabels(labels)
    remove_classes = []

    for i in range(len(classesInLabels(labels))): 
        if(not(classes[i] in seperation)): 
            remove_classes.append(classes[i])
            

    for i in range(len(remove_classes)):
        if(remove_classes[i] in labels):
            train_data = train_data[train_labels != remove_classes[i]]
            train_labels = train_labels[train_labels != remove_classes[i]]
        
    return train_data, train_labels

def classesSvmBranches(svm_tree_branch):

    """
    Collects the classes every svm model is trained in a SVMBDT. 
    Returns the layers of branches. 
    """

    cSB = [svm_tree_branch[0].classes_]
    for i in range(1,len(svm_tree_branch)): 
        branch = []
        for j in range(len(svm_tree_branch[i])): 
            if(not(svm_tree_branch[i][j] == False)): 
                branch.append(svm_tree_branch[i][j].classes_)
        cSB.append(branch)
    
    return cSB

def SupportVectorsSvmBranches(svm_tree_branch):

    """
    Collects the amount of Support Vectors in each SVM in a SVMBDT. 
    Return an array the amount for each branch. 
    """

    cSB = []
    cSB.append(svm_tree_branch[0].n_support_)
    for i in range(1,len(svm_tree_branch)): 
        branch = []
        for j in range(len(svm_tree_branch[i])): 
            if(not(svm_tree_branch[i][j] == False)): 
                branch.append(svm_tree_branch[i][j].n_support_)
        cSB.append(branch)
    
    return cSB

def TotalAndMeanSupportVectors(support_vectors_array): 

    """
    Returns a the total and mean support vectors for one SVMBDT design. 
    """

    all_support_vectors = []
    for i in range(len(support_vectors_array)): 
        for j in range(len(support_vectors_array[i])):
                try:
                    all_support_vectors.append(support_vectors_array[i][j][0])
                    all_support_vectors.append(support_vectors_array[i][j][1])
                except: 
                    all_support_vectors.append(support_vectors_array[i][j])

    total_support_vector = np.sum(all_support_vectors)
    mean_support_vector = np.mean(all_support_vectors)     

    return total_support_vector, mean_support_vector

def MeanAccuracy(data_and_labels_branch, sub_predicted_labels):


        """
        Calculates the mean accuracy with the accuracy for every class. 
        Return the mean_accuracy. 
        
        """

        mean_accuracy = []

        accuracy = sum(data_and_labels_branch[0][1] == sub_predicted_labels[0])/len(data_and_labels_branch[0][1])

        mean_accuracy.append(accuracy)

        for i in range(1,len(data_and_labels_branch)-1):  
            for j in range(len(data_and_labels_branch[i])): 
                if(len(data_and_labels_branch[i][j][1]) < len(sub_predicted_labels[i][j])): 
                    buffer_data = data_and_labels_branch[i][j][1][:len(sub_predicted_labels[i][j])]
                    accuracy = sum(buffer_data == sub_predicted_labels[i][j])/len(buffer_data)
                    mean_accuracy.append(accuracy)
                
                elif(len(data_and_labels_branch[i][j][1]) > len(sub_predicted_labels[i][j])): 
                    buffer_data = sub_predicted_labels[i][j][1][:len(data_and_labels_branch[i][j][1])]
                    accuracy = sum(buffer_data == data_and_labels_branch[i][j][1])/len(buffer_data)
                    mean_accuracy.append(accuracy)

                else: 
                    accuracy = sum(sub_predicted_labels[i][j] == data_and_labels_branch[i][i][j])/len(sub_predicted_labels[i][j])
                    mean_accuracy.append(accuracy)

        return np.mean(mean_accuracy)

def CombineMultiBranch(sub_branch_label,sub_tree_branch, branch_label): 

    """
    Combines a sub-tree to a branch in a SVMBDT. 
    The minimum class in the branch are replaced with the new label from the sub-tree. 
    Return the combine_labels that must be placed in the tree branch.
    
    """
    
    combined_sub_label = copy.deepcopy(sub_branch_label)
    combined_label = copy.deepcopy(branch_label)
    
    classes = sub_tree_branch[0][0] + sub_tree_branch[0][1]

    combined_label[combined_label == min(classes)] = combined_sub_label

    return combined_label


def SvmDesionTreeTrain(train_data, train_labels, tree_branches, svm_branch_models = {}, sub_data = 0):

    """
    Takes in the train_data, train_labels, tree_branches and svm_branch_models 
        - 
        - Uses the BranchDataLabels to divide the different classes in the train_data and train labels picture. 
        - tree_branches structure is a dictornary form of:
            - {0 : [[A_0,...,A_n],[B_0,...,B_n]], 1 : [[[A_0],[A_1,...,A_n]],[[B_0],[B_1,...,B_m]]], ... , N : {[[[A_(n-1)],[A_n]],[[B_(m-1)],[B_m]]]}
        - the svm_branch_models strucutre is a dictonary form of: 
            - {0 : SVC(), 1 : [SVC(), SVC()], ..., N : [SVC(),SVC()]}
            - If unsymmetric the induvial branch example: [False, SVC()] 

    Prints: the time of training for the whole svm_branch_models. 

    Returns svm_branch_models 
        - for also return data_and_labels_branches, sub_data = 1. 

    """

    import time

    time_start = time.time()

    data_and_labels_branches = BranchDataLabels(train_data, train_labels, tree_branches)


    if(len(svm_branch_models) > 0): 
        svm_branch_models = SvmBranchModelTrain(data_and_labels_branches, svm_branch_models)

    time_stop = time.time()

    training_time = time_stop - time_start

    print("The training time is: " + str(round(training_time,3)) + str(" sec."))

    if(sub_data):
        return svm_branch_models, data_and_labels_branches
    
    else: 
        return svm_branch_models
    

def SvmDesionTreePredict(test_data, svm_branch_models, tree_branches, sub_data = 0): 

    """
    Takes in test_data, svm_branch_models and tree_branches. The svm_branch_models contains of
    SVM Models that are trained form the SvmDesiionTreeTrain() functions. The format of 
    svm_branch_models and tree_branches can be found in the SvmDFesionTreeTrain().
    The prediction time is printed to console.  
    - Returns the predicted labels.
        - If sub_data_return = 1; return predicted_label, sub_data_branch and predicted_branch_labels. 
    
    """
    
    time_start = time.time()

    if(sub_data): 
        predicted_branch_labels, sub_data_branch = SvmBranchModelPredict(test_data, svm_branch_models, tree_branches, sub_data_return=sub_data)
    
    else: 
        predicted_branch_labels= SvmBranchModelPredict(test_data, svm_branch_models, tree_branches)

    predicted_label = CombineLabels(predicted_branch_labels, tree_branches)

    time_stop = time.time()

    prediction_time = time_stop - time_start

    print("The prediction time is: " + str(round(prediction_time,3)) + str(" sec."))

    if(sub_data): 
        return predicted_label, sub_data_branch, predicted_branch_labels   
    else: 
        return predicted_label