# data_generator.py
# Creating datset of vector inputs for NN
# Author: maria.olaru@

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Input preferences
N = 2   # number of vectors
L = 100  # length of datapoints 

def generate_vectors():
    """ Create speech vectors. """
    np.random.seed(20)
    v = np.random.uniform(0, 1, N*2)
    v = v.reshape(N, 2)
    return v
        
def plot_vectors():
    """ Plots speech vectors. """
    sns.set(style="white", palette="muted", color_codes=True)
    color_array = sns.color_palette("hls", N)
    
    fig, ax = plt.subplots()
    
    origin = np.zeros(N)
    v = generate_vectors()
    print("vector: \n", v)
    
    ax.quiver(origin, 
              origin, 
              v[:,0], 
              v[:,1], 
              color = color_array, 
              scale = 2)
    ax.axis([0, 1, 0, 1])
    
    plt.title("Speech Vectors")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    
    plt.show()

def generate_data():
    """ Create datset. """
    d = np.zeros((N, L, 2)) 
    v = generate_vectors()   
    for i in range(N):
        d[i, 0:L, 0] = np.linspace(0, v[i, 0], L)
        d[i, 0:L, 1] = np.linspace(0, v[i, 1], L)
    return d

def process_data():
    """ Converge & transform to 1D vector. """
    d = generate_data()
    
    
    indx_s = np.random.permutation(d.shape[0]) #shuffle vector type
    ds = d[indx_s, :, :]
    dsv = ds.reshape(-1, order = 'F') #output col1 (dim1) then col2 (dim2)
    
    return dsv

def main():
    plot_vectors()
 
    fn_begin = 'lang_data_v1.0_' + str(N) + '_' + str(L) + '_'
    np.savetxt(fn_begin + 'input.csv', process_data(), 
               delimiter = ',')

if __name__ == '__main__': 
    main()