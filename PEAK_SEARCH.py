import numpy as np
import pandas as pd

class schema:
    data = "data"
    freq_ = "freq"
    freq = 'freq_data' 
    trans = 'trans_data' 
    phase = 'phase_data' 
    realdata = 'real_data' 
    imagdata = 'imag_data' 
    freqfit = 'freq_fit' 
    transfit = 'trans_fit' 
    realfit = 'real_fit' 
    imagfit ='imag_fit' 
    n = 'fit_n' 
    loss = 'fit_los' 
    nerr = 'fit_n_err'
    losserr = 'fit_los_err'
    transdiff = 'trans_diff'

def importfile(file):
    f = np.load(file)
    freq = f[schema.freq]
    data = f[schema.trans]
    
    df = pd.DataFrame({schema.freq: freq, 
                       schema.trans: data})
    return df