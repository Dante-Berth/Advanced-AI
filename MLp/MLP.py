import scipy.stats
import pywt
from pywt import wavedec
import numpy as np
import pywt
import pandas as pd
#Chargement

dataset = pd.read_csv(PATH,compression='zip')
dataset = dataset.drop(dataset.columns[0],axis=1)
dataset["open_date"] = pd.to_datetime(dataset["open_time"], unit='ms')
a = 22000
b = 28000


def denoising_signal(name,base,level,details):
    coeffs = wavedec(dataset[name].to_numpy(), base, level=level)
    c=0
    for i in range(1,details):
        a = len(coeffs[-i])
        coeffs[-i] = np.zeros(a)
        c = c + a
    rebuilt_signal = pywt.waverec(coeffs, base)

# Example usage
a = 22000
b = 28000
liste_values = dataset[a:b]["open_price"].to_numpy()
window_size = 50
denoised_values = denoise_signal(liste_values, window_size)

# Print the denoised values
print(denoised_values)

