from sklearn import svm 
import scipy.io
import numpy as np
from spectral import *

def create_hyper_image(dataset): 
    #Insperation from: https://www.youtube.com/watch?v=Yu6NrxiV91Y
    img = open_image(dataset)

def write_dataset_to_file(dataset, filename):
    f = open(filename)
    for i in range(len(dataset)): 
        f.write(str(dataset[0]))
    f.close()





