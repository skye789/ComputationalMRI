
import numpy as np
import pywt


def HardT(x,t):
    return x * (np.abs(x) > t);

def SoftT(x,t):
    return x/np.abs(x) * np.maximum(np.abs(x) - t, 0);

def complex_wd2(x):
    c_real = pywt.wavedec2(np.real(x), 'db4', mode='periodization', level=4)  #wavelet decomposition
    c_imag = pywt.wavedec2(np.imag(x), 'db4', mode='periodization', level=4)
    c_real_array, s = pywt.coeffs_to_array(c_real)
    c_imag_array, _ = pywt.coeffs_to_array(c_imag)
    
    return c_real_array + 1j * c_imag_array, s

def complex_wr2(x,s):
    c_real_thresh = pywt.array_to_coeffs(np.real(x), s, output_format='wavedec2')
    c_imag_thresh = pywt.array_to_coeffs(np.imag(x), s, output_format='wavedec2')
    rec_real = pywt.waverec2(c_real_thresh, 'db4', mode='periodization')            #wavelet reconstruction
    rec_imag = pywt.waverec2(c_imag_thresh, 'db4', mode='periodization')

    return rec_real + 1j * rec_imag

