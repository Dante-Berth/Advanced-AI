import pywt
import numpy as np

def denoise_signal(signal, window_size, wavelet):
    # Create an empty array to store denoised signal
    denoised_signal = np.zeros_like(signal)

    # Iterate over each time point
    for t in range(len(signal)):
        # Define the start and end indices of the window
        start = max(0, t - window_size)
        end = t + 1

        # Extract the windowed portion of the signal
        window = signal[start:end]

        # Apply wavelet-based denoising
        coeffs = pywt.wavedec(window, wavelet)
        threshold = np.sqrt(2 * np.log(len(window)))  # Adjust the threshold as needed
        denoised_coeffs = [pywt.threshold(c, threshold) for c in coeffs]
        denoised_window = pywt.waverec(denoised_coeffs, wavelet)

        # Update the denoised signal with the windowed denoised portion
        denoised_signal[t] = denoised_window[-1]

    return denoised_signal

# Define your input signal
signal = [0.5, 1.2, 2.1, 3.7, 4.8, 2.9, 1.6, 0.8, 1.5, 2.3, 4.0, 3.2, 2.1]
window_size = 3
wavelet = 'db4'
denoised = denoise_signal(signal, window_size, wavelet)
print(denoised)


def denoising_signal_with_window(signal, base, level, details, window_size):
    """
    Function that denoises the signal using PyWavelets and a slicing window.
    """
    signal_length = len(signal)
    denoised_signal = np.zeros(signal_length)

    for i in range(signal_length):
        # Define the window indices
        start_idx = max(0, i - window_size)
        end_idx = min(signal_length, i + window_size + 1)

        # Extract the windowed signal
        windowed_signal = signal[start_idx:end_idx]

        # Decompose the windowed signal into wavelet coefficients
        coeffs = pywt.wavedec(windowed_signal, base, level=level)

        # Determine the number of coefficients to keep
        num_coeffs_to_keep = len(coeffs) - details

        # Set the unwanted detail coefficients to zero
        for j in range(1, details + 1):
            coeffs[-j] = np.zeros_like(coeffs[-j])

        # Reconstruct the denoised windowed signal
        denoised_window = pywt.waverec(coeffs[:num_coeffs_to_keep], base)

        # Update the denoised signal with the windowed denoised signal
        denoised_signal[i] = denoised_window[window_size]
        print(start_idx, end_idx,i)
    return denoised_signal
denoised_signal = denoising_signal_with_window(signal, 'db4', 1, 1, window_size=5)
print(denoised_signal)
