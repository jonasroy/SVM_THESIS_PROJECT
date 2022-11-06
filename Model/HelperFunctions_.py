from sklearn import svm 
import scipy.io
import numpy as np
import os 
import copy
from sklearn import preprocessing

def create_hyper_image(dataset): 
    #Insperation from: https://www.youtube.com/watch?v=Yu6NrxiV91Y
    img = open_image(dataset)

def write_dataset_to_file(dataset, filename):
    f = open(filename)
    for i in range(len(dataset)): 
        f.write(str(dataset[0]))
    f.close()

def split_dataset_paths(dir_path = "", file_types = []):
    splited_paths = {}
    paths = os.listdir(dir_path)   
    
    for types in file_types: 
        splited_paths[types] = []
    
    for file in range(len(paths)): 
        for type in file_types: 
            if(type in paths[file]): 
                splited_paths[type].append(dir_path + paths[file])

    return splited_paths 

def OneClass(labels, class_values = []): 
    """
      Just includes the class values from the class_values array. 
      Replaces the class value with new number with same order. 
      Makes ClassesVsRest. class_values not includer are the rest class. 

    """
    class_values = np.array(class_values)

    new_class_values = []

    for i in range(len(class_values)): 
        new_class_values.append(i)

    new_labels = copy.deepcopy(labels)

    for i in range(len(labels)): 
        for j in range(len(labels[0])): 
                if(new_labels[i][j] in class_values): 
                    class_position = np.where(class_values == new_labels[i][j])
                    new_labels[i][j] = new_class_values[class_position[0][0]]
                else: 
                    new_labels[i][j] = len(class_values)

    return new_labels


def subFrame(data, labels, x1, x2, y1, y2): 

    """
    Make a subframe of a dataset picture and 
    returns sub data and label. 
    ROIs (region of interests)
    """

    sub_data = []
    sub_label = []
    n_data_bands = len(data[0])

    for i in range(y1,y2): 
        sub_label.append(labels[i][x1:x2])
        x_row = []
        for j in range(n_data_bands): 
            x_row.append(data[i][j][x1:x2])
        sub_data.append(x_row)

    return np.array(sub_data), np.array(sub_label)
    


def preprocesing_data(data,labels): 
    # prepare data
    X = data.transpose(0,2,1).reshape((-1,61))
    y = labels.flatten()

    X = preprocessing.scale(X, axis=0)                    # Normalization
    #X = preprocessing.normalize(X, axis=0) 
    return X,y 


def lessBands(data, n_band_dim):
    import random 

    x_dim = len(data[0])
    y_dim = len(data)
    n_bands = len(data[0][0])

    random_bands = []
    while(len(random_bands) <= n_band_dim):  
        random_band = random.randint(0,n_bands-1)
        if((random_band in random_bands) == False): 
            random_bands.append(random_band)

    new_data = []

    for i in range(y_dim):
        x_row = [] 
        for j in range(x_dim): 
            pixel = []
            for k in range(n_band_dim):
                pixel.append(data[i][j][random_bands[k]])
            x_row.append(pixel)
        new_data.append(x_row)
    
    return np.array(new_data)

            
def combinePictures(data1, labels1, data2, labels2): 
    """

    The index of data and the lables must correspond
    Must be same dimensions
    
    """
    combinded_data = []
    combinded_labels = []

    for i in range(len(data1)): 
        combinded_data.append(data1[i])
    for i in range(len(data2)): 
        combinded_data.append(data2[i])
    
    for i in range(len(labels1)): 
        combinded_labels.append(labels1[i])
    
    for i in range(len(labels2)): 
        combinded_labels.append(labels2[i])

    return combinded_data, combinded_labels


def combineModels(models, final_estimator): 

    """
    Combine multiple sklearn models in too one.  
    """

    from sklearn.ensemble import StackingClassifier
    stack_models = StackingClassifier(estimators=models)
    return stack_models

def saveModel(model, path):
    "Should be saved as 'model_name.sav'"
    import pickle 
    pickle.dump(model, open(path, "wb"))

def loadModel(path): 
    import pickle 
    return pickle.load(path, "rb")
    
def confusionMatrix(data, labels):
    Xout = stack_models.predict(X)

    plt.imshow(Xout.reshape((200,200)))

    cm = confusion_matrix(labels.flatten(), data, normalize='true')

    plt.imshow(cm, vmax=1, vmin=0) 
    plt.ylabel('True')
    plt.xlabel('prediction')
    plt.colorbar()

def namesOnLabels(labels, labels_name):
    labels_with_names = []
    for i in range(len(labels)): 
        X_labels = []
        for j in range(len(labels[0])):
            X_labels.append(labels_name[labels[i][j]])
        labels_with_names.append(X_labels)  
    return np.array(labels_with_names)

def classesInLabels(labels): 
    try: 
        classes_ = []
        for i in range(len(labels)): 
            for j in range(len(labels[0])):
                if(not(labels[i][j] in classes_)): 
                    classes_.append(labels[i][j])
        return np.sort(classes_)
    except: 
        classes = []
        for i in range(len(labels)): 
             if(not(labels[i] in classes_)): 
                    classes_.append(labels[i])
        return np.sort(classes_) 

def combineLabelClasses(labels, combining_classes): 
    """
    Combine the lables. example the three species and water. 
    """
    new_labels = []
    labels_classes = classesInLabels(labels)
    
    new_class = combining_classes[0]

    new_classes = []

    for i in range(len(labels_classes)):
        if(labels_classes[i] in combining_classes): 
            new_classes.append(new_class)
        else: 
            if(labels_classes[i] < new_class):
                new_classes.append(labels_classes[i])
            else: 
                new_classes.append(labels_classes[i] - len(combining_classes) + 1)
        
    for i in range(len(labels)): 
        x_labels = []
        for j in range(len(labels[0])): 
            x_labels.append(new_classes[labels[i][j]])
        new_labels.append(x_labels)

    return np.array(new_labels)


def flipData(data): 

    data_flipped = []


    for i in range(len(data[0])): 
        x_data = []
        for j in range(len(data)): 
            x_data.append(data[j][i])
        data_flipped.append(x_data)

    return np.array(data_flipped)


def reshape_sj(data): 
    """
    Used to reshape the datasets samson and jasper.
    """

    reshaped_data_buffer = []
    pixel_dim = int(np.sqrt(len(data[0])))

    for i in range(len(data)):
        reshaped_data_buffer.append(np.reshape(np.array(data[i]), (pixel_dim, pixel_dim)))

    reshaped_data = []

    for i in range(len(reshaped_data_buffer[0])): 
        X_data = []
        for j in range(len(reshaped_data_buffer)): 
            X_data.append(reshaped_data_buffer[j][i])
        reshaped_data.append(X_data)

    return np.array(reshaped_data)


def SeperateDataLabels(data,labels, seperation): 

    train_data = copy.deepcopy(data)
    train_labels = copy.deepcopy(labels)

    #Check the labels and the seperation have the same number of classes or not 
    # Will remove the classes not in the seperation array. 
    
    if(len(seperation) == 2): 

        classesInSeperation = []

        for i in range(len(seperation)): 
            for j in range(len(seperation[i])): 
                if(not(seperation[i][j] in classesInSeperation)): 
                    classesInSeperation.append(seperation[i][j])

        classesInSeperation.sort()
        classesInSeperation = np.array(classesInSeperation)
        cIL = classesInLabels(labels)

        #Checks if the seperation and the labels have same amount of classes. If not return -1 ????
        if(not((classesInSeperation == cIL).all())): 

            for i in range(len(cIL)): 
                if(not(cIL[i] in classesInSeperation)):
                    print(cIL[i])
                    train_data = train_data[train_labels != cIL[i]]
                    train_labels = train_labels[train_labels != cIL[i]]

        for i in range(len(seperation)): 
            for j in range(len(seperation[i])):
                if(seperation[i][j] != min(seperation[i])):
                    
                    train_labels[train_labels == seperation[i][j]] = min(seperation[i])
        
        return train_data, train_labels
        
    else: 
        print("The seperation needs to be 2 in length. For 1v1 classification")
        return -1


#seperation = {0 : [[1,2,3], [4,5,6]] , 1 : [[[1,2], [3]],[[4], [5,6]]]}

def SvmDesionTree(data, labels, tree_branches):
    """
    Every branch in the SVM Desion Tree is aimed to be 1v1 classification. 
    """

    #n_svm_models = len(tree_branches[len(tree_branches)]) + 1 #Need to find out the amount of models to be added to array 
    svm_models = {}
    data_and_labels_branches = {}
    
    #Stores the different training and labeling for the different branches
    #Every layer are stored in data_and_labels_branches dictornary and are in coresponding placement as tree_branches dictonary. 

    if(len(tree_branches) > 0): 
        for i in range(0,len(tree_branches)): 
            data_and_labels = []
            try: 
                for j in range(len(tree_branches[i])): 
                    data_and_labels.append(SeperateDataLabels(data, labels, tree_branches[i][j]))
            except: 
                data_and_labels = SeperateDataLabels(data, labels, tree_branches[i])

            data_and_labels_branches[i] = data_and_labels
    #Train all the SVM Models for the different tree branches from the data and labeling in data_and_labels_branches.
    # Stores the predicted labels in the predict_labels dictonary

    predicted_labels = {} 

    print(data_and_labels_branches)

    #for i in range(len(data_and_labels_branches)): 
    #    for j in range(len(data_and_labels_branches[i])): 

    """
            train_data, train_labels = data_and_labels_branches[i]
            
            svm = SVC()
            svm.fit(train_data, train_labels)
            yout = svm.predict(train_data) 

            svm_models[i].append(svm)  
            predicted_labels[i].append(yout)
    """
    #Combine all the SVM to one labeling.

   # print(predicted_labels)
    

    return 0 

def combineLabels(predicted_labels): 
    """
    Combine the result from the desion tree. Should be automated. 
    """
    
    for i in range(len(predicted_labels), 0): 
        for j in range(len(predicted_labels)): 
            branch1 = []
            branch2 = []

            if(predicted_labels[i][j] == 2):  


    


    return 0 



def BranchDataLabels(data, labels, tree_branches): 

    """
    Takes in data and labes for the SVM Desion Tree. 
    Makes the labels for every branch 1v1 classes. 

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


def SvmBranchModelTrain(data_and_labels_branches): 

    svm_branch_models = {}

    svm = SVC()
    train_data, train_labels = data_and_labels_branches[0]
    svm.fit(train_data, train_labels)
    svm_branch_models[0] = svm

    for i in range(1,len(data_and_labels_branches)): 
        svm_models = []
        for j in range(len(data_and_labels_branches[i])): 
            if(len(data_and_labels_branches[i][j]) == 2): 
                train_data, train_labels = data_and_labels_branches[i][j]
                svm = SVC()
                svm.fit(train_data, train_labels)
                svm_models.append(svm)
            else: 
                svm_models.append(False)
        svm_branch_models[i] = svm_models

    return svm_branch_models


def SvmBranchModelPredict(data_and_labels_branches, svm_branch_models): 
    predicted_branch_labels = {}

    svm = svm_branch_models[0]
    train_data, train_labels = data_and_labels_branches[0]
    yout = svm.predict(train_data)
    predicted_branch_labels[0] = yout

    for i in range(1,len(data_and_labels_branches)): 
        predicted_labels = []
        for j in range(len(data_and_labels_branches[i])): 
            if(len(data_and_labels_branches[i][j]) == 2): 
                train_data, train_labels = data_and_labels_branches[i][j]
                svm = svm_branch_models[i][j]
                yout = svm.predict(train_data)
                predicted_labels.append(yout)
            else: 
                predicted_labels.append([])
        predicted_branch_labels[i] = predicted_labels

    return predicted_branch_labels
