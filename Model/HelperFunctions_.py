from sklearn import svm 
import scipy.io
import numpy as np
import os 
import copy

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

def OneClass(labeled_data, class_value=0): 
    # Not changing original data 
    lb = [[]*len(labeled_data)]
    for i in range(len(labeled_data)): 
        for j in range(len(labeled_data[0])):
             lb[i].append(labeled_data[i][j])
    #Making the labeled data just the one class. 
    for i in range(len(ld)): 
        for j in range(len(ld[0])): 
            if(ld[i][j] != class_value): 
                ld[i][j] = 9

