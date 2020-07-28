# data_generator.py
# Generate two synthetic signal timeseries and mix the signals
# using a 2 x 2 weight matrix
# Author: maria.olaru@

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import signal


# Input preferences
T = np.linspace(0, 1, 800, endpoint = False) #Sampling at 800Hz for 1s
W = np.matrix('4 2; 2 4') # Weights

def generate_signals():
    """ Create a sine and square signal """    
    sig1 = np.sin(2 * np.pi * 10 * T)
    sig2 = np.sin(2 * np.pi * 5 * T)
    #sig2 = signal.square(2 * np.pi * 10 * T) Square wave
    
    sig_vec = np.column_stack((sig1, sig2))
    
    return sig_vec
        
def plot_signals():
    """ Plots speech signals. """
    sig_vec = generate_signals()
    plt.plot(T, sig_vec[:,0], label = "sine wave 1")
    plt.plot(T, sig_vec[:,1], label = "sine wave 2")
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Synthetic Sound Signal')
    
    plt.legend()
    plt.show()

def mix_signals():
    """ Multiply incoming signal (Nx2) by weighting matrix (2x2) 
          and output a Nx2 mixed signal """

    sig_vec = generate_signals()      
    M = sig_vec * W

    print(M)
    
    return M

def main():
    plot_signals()
    
    #Save original signal
    fn_begin = 'data/lang_input_unmixed'
    np.savetxt(fn_begin + '.csv', generate_signals(), 
               delimiter = ',')
    
    #Save weights
    fn_begin = 'data/lang_input_weights'
    np.savetxt(fn_begin + '.csv', generate_signals(), 
               delimiter = ',')
    
    #Save mixed signal
    fn_begin = 'data/lang_input_mixed'
    np.savetxt(fn_begin + '.csv', W, 
               delimiter = ',')

if __name__ == '__main__': 
    main()