"""
HelperFunctions_.py: 

Helper functions used for the SVMBDT project.
They are used for preproccesing, gather infromation from the SVMBDT and test different ML Functions. 

"""

__author__ = "Jonas Gjendem Røysland"
__email__ = "jonasroy@stud.ntnu.no"
__date__ = "2022/12/18"
__version__ = "1.0.0"

import numpy as np
import os 
import copy
from sklearn import preprocessing
import time
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


def split_dataset_paths(dir_path = "", file_types = []):

    """
    Collects the file names in a array. 
    The functions returns splited_paths. 
    """

    splited_paths = {}
    paths = os.listdir(dir_path)   
    
    for types in file_types: 
        splited_paths[types] = []
    
    for file in range(len(paths)): 
        for type in file_types: 
            if(type in paths[file]): 
                splited_paths[type].append(dir_path + paths[file])

    return splited_paths 

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


def preprocessing_data(data,labels, shape): 
    """
    Preproccesing the data and labels from a Hyperspectral Image and its Ground Truth. 
    The shape is the amount of spectral bands in the data. 
    Returns X,y
    """

    X = data.transpose(0,2,1).reshape((-1,shape))
    y = labels.flatten()

    X = preprocessing.scale(X, axis=0)                
    #X = preprocessing.normalize(X, axis=0) 
    return X,y 


            
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



def classesInLabels(labels): 

    """
    Collects the classes_ in a label and store it
    in an array. 
    Returns the classes_ in the labels. 
    """

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
    """
    Used to flip the data image 90 degress. 
    Returns the data_flipped. 
    """

    data_flipped = []


    for i in range(len(data[0])): 
        x_data = []
        for j in range(len(data)): 
            x_data.append(data[j][i])
        data_flipped.append(x_data)

    return np.array(data_flipped)


def reshape_sj(data): 
    """
    Used to reshape the dataset Samson and Jasper Ridge. 
    Return the reshaped_data. 
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

def classesSvmBranches(svm_tree_branch):
    cSB = [svm_tree_branch[0].classes_]
    for i in range(1,len(svm_tree_branch)): 
        branch = []
        for j in range(len(svm_tree_branch[i])): 
            if(not(svm_tree_branch[i][j] == False)): 
                branch.append(svm_tree_branch[i][j].classes_)
        cSB.append(branch)
    
    return cSB

def SupportVectorsSvmBranches(svm_tree_branch):
    cSB = [svm_tree_branch[0].n_support_]
    for i in range(1,len(svm_tree_branch)): 
        branch = []
        for j in range(len(svm_tree_branch[i])): 
            if(not(svm_tree_branch[i][j] == False)): 
                branch.append(svm_tree_branch[i][j].n_support_)
        cSB.append(branch)
    
    return cSB


def newColorLabels(labels, dimx, dimy, color_label): 

    """
    Define a new color scheme for the labels. 
    Each classes are defined with the RGB code in hex, 0-255.
    Color labeling is defined by: color_label = {0 : [r,g,b], ..., N : [r,g,b]}, 
    with N amount of classes. 

    Returns the new_new_label with a new color scheme. 
    """

    new_label = copy.deepcopy(labels)
    new_label = new_label.reshape(dimx, dimy)
    new_new_label = []

    for i in range(len(new_label)): 
        x_array = []
        for j in range(len(new_label[0])): 
            x_array.append(color_label[new_label[i][j]])
        new_new_label.append(x_array)
    
    return np.array(new_new_label)


def SingleMachineLearningTest(data, label, train_data, train_label, iter, kmeans_cluster_number): 

    """
    Test these different machine learning (ML) algorithms: 
        - SVM Linear One-vs-One 
        - SVM Linear One-vs-Rest
        - SVM RBF(Radial Basis Function)
        - KMeans
        - KNearestNeighbor 
        - Random Forest

    Each function is provided by the library Sklearn. 
    The training time, prediction time and overall accuracy are printed to console. 

    - Returns the labeled data for each of the ML algortihms. 
    """

    svm_linear_ovo = SVC(kernel="linear",class_weight= "balanced", max_iter=iter, decision_function_shape="ovo")
    svm_linear_ovr = LinearSVC(class_weight= "balanced", max_iter=iter)
    svm_rbf = SVC(kernel="rbf",class_weight= "balanced", max_iter=iter, decision_function_shape="ovr")
    k_means = KMeans(n_clusters=kmeans_cluster_number, random_state=0)
    k_nearest_neighbor = KNeighborsClassifier(n_neighbors=len(classesInLabels(label)-2))
    random_forest = RandomForestClassifier()

    #Train the different models with printing of training time. 

    start_time = time.time()

    svm_linear_ovo.fit(train_data, train_label)

    stop_time = time.time() 

    print("Linear 1-vs-1 Training Time: " + str(round(stop_time - start_time, 3)) + "sec.")


    start_time = time.time()

    svm_linear_ovr.fit(train_data, train_label)

    stop_time = time.time() 

    print("Linear 1-vs-Rest Training Time: " + str(round(stop_time - start_time, 3)) + "sec.")


    start_time = time.time()

    svm_rbf.fit(train_data, train_label)

    stop_time = time.time() 

    print("RBF Training Time: " + str(round(stop_time - start_time, 6)) + "sec.")


    start_time = time.time()

    k_means.fit(train_data, train_label)

    stop_time = time.time() 

    print("Kmeans Training Time: " + str(round(stop_time - start_time, 3)) + "sec.")

    
    start_time = time.time()

    k_nearest_neighbor.fit(train_data, train_label)

    stop_time = time.time() 

    print("KNearestNeighbor Training Time: " + str(round(stop_time - start_time, 3)) + "sec.")


    start_time = time.time()

    random_forest.fit(train_data, train_label)

    stop_time = time.time() 

    print("Random Forest Training Time: " + str(round(stop_time - start_time, 3)) + "sec.")

    
    #Predict the different models with time. 

    start_time = time.time()

    linear_ovo_yout = svm_linear_ovo.predict(data)

    stop_time = time.time()

    print("Linear 1-vs-1 Predict Time: " + str(round(stop_time - start_time, 3)) + "sec.")


    start_time = time.time()

    linear_ovr_yout = svm_linear_ovr.predict(data)

    stop_time = time.time()

    print("Linear 1-vs-Rest Predict Time: " + str(round(stop_time - start_time, 3)) + "sec.")


    start_time = time.time()

    rbf_yout = svm_rbf.predict(data)

    stop_time = time.time()

    print("RBF Predict Time: " + str(round(stop_time - start_time, 3)) + "sec.")

    
    start_time = time.time()

    kmeans_yout = k_means.predict(data)

    stop_time = time.time()

    print("KMeans Predict Time: " + str(round(stop_time - start_time, 3)) + "sec.")


    start_time = time.time()

    knearest_yout = k_nearest_neighbor.predict(data)

    stop_time = time.time()

    print("KNearestNeighbor Predict Time: " + str(round(stop_time - start_time, 3)) + "sec.")


    start_time = time.time()

    random_forest_yout = random_forest.predict(data)

    stop_time = time.time()

    print("Random Forest Predict Time: " + str(round(stop_time - start_time, 3)) + "sec.")


    #The overall accuracy and support vectors

    print("Linear 1-vs-1 Accuracy: " + str(100*round(sum(linear_ovo_yout == label)/len(label),4)) + "%")
    print("Linear 1-vs-Rest Accuracy: " + str(100*round(sum(linear_ovr_yout == label)/len(label),4)) + "%")
    print("RBF Accuracy: " + str(100*round(sum(rbf_yout == label)/len(label),4)) + "%" )
    print("KMeans Accuracy: " + str(100*round(sum(kmeans_yout == label)/len(label),4)) + "%")
    print("KNearestNeighbour Accuracy: " + str(100*round(sum(knearest_yout == label)/len(label),3)) + "%")
    print("Random Forest Accuracy: " + str(100*round(sum(random_forest_yout == label)/len(label),3)) + "%")

    print("Linear 1-vs-1 Total SVM Support_Vectors : " + str(sum(svm_linear_ovo.n_support_)))
    print("RBF Total SVM Support_Vectors : " + str(sum(svm_rbf.n_support_)))

    print("Linear 1-vs-1 Average SVM Support_Vectors : " + str(round(np.mean(svm_linear_ovo.n_support_),3)))
    print("RBF Average SVM Support_Vectors : " + str(round(np.mean(svm_rbf.n_support_),3)))
    

    return linear_ovo_yout, linear_ovr_yout, rbf_yout, kmeans_yout, knearest_yout, random_forest_yout


