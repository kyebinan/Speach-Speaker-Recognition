import numpy as np
import scipy
from Lab1.lab1_tools import *
from scipy.signal import lfilter
from scipy.signal import hamming
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
import matplotlib.pyplot as plt



# DT2119, Lab 1 Feature Extraction

# Function given by the exercise ----------------------------------

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        samples: speech samples
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    # We have a length of 16 640 frames, we want to split these into N windows, where each window has a length of winlen.
    # The window sliding step is winshift, meaning we will have some overlap in the windows.
    windows = []

    p = 0
    while (p < len(samples) - winlen):  # While we can still form a whole window
      windows.append(samples[p:p+winlen])
      p += winshift
    return np.array(windows)
    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    # Filter coefficients
    b = [1, -p]  # Numerator coefficients
    a = [1]      # Denominator coefficients, indicating a FIR filter

    # Apply the pre-emphasis filter
    output = scipy.signal.lfilter(b, a, input)
    return output

def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    M = input.shape[1]

    # Generate the Hamming window with the same number of samples as in a frame
    # Using sym=False to generate the periodic version of the Hamming window
    # which is more suitable for spectral analysis
    window = scipy.signal.windows.hamming(M, sym=False)

    # Apply the window to each frame
    # This is done by element-wise multiplication of each frame by the window
    output = input * window
    return output

def powerSpectrum(input, nfft=512):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    # Apply FFT on each frame and take the first nfft components
    fft_result = scipy.fftpack.fft(input, n=nfft)

    # Calculate the power spectrum for each frame
    # This is done by taking the square of the magnitude of the FFT result
    # np.abs(fft_result) computes the magnitude (sqrt(re^2 + im^2))
    # Squaring the magnitude gives us the power spectrum
    output = np.abs(fft_result) ** 2

    return output

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    # Calculate the Mel filterbank
    mel_filterbank  = trfbank(fs=samplingrate, nfft=input.shape[1])

    # Apply the Mel filterbank to the power spectrum (matrix multiplication)
    mel_spectrum = np.dot(input, mel_filterbank.T)

    # Compute the logarithm of the Mel spectrum
    log_mel_spectrum = np.log(mel_spectrum + np.finfo(float).eps)

    return log_mel_spectrum

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    # from the lab description seems like putting n=nceps will affect the output
    # so I instead pick them out only after the computation is complete.
    return scipy.fftpack.dct(x=input)[:, 0:nceps]

def dtw(local_distances, plot_best_path=False):
    """
    Perform Dynamic Time Warping (DTW) on a matrix of local distances between two sequences.
    Optionally plots the best path of alignment.

    Args:
        local_distances (np.ndarray): A 2D NumPy array of shape [N, M], where N is the length
            of the first sequence, M is the length of the second sequence.
        plot_best_path (bool): If True, the function will plot the best alignment path over
            the matrix of local distances.

    Returns:
        float: The normalized global distance between the two sequences.
    """
    N, M = local_distances.shape
    AD = np.full((N+1, M+1), np.inf)  # Accumulated distance matrix
    AD[0, 0] = 0

    # Populate the accumulated distance matrix
    for i in range(1, N+1):
        for j in range(1, M+1):
            cost = local_distances[i-1, j-1]
            AD[i, j] = cost + min(AD[i-1, j], AD[i, j-1], AD[i-1, j-1])
    
    d = AD[N, M] / (N + M)  # Normalized global distance

    # Plot the best path if requested
    if plot_best_path:
        path = []
        i, j = N, M
        while i > 0 or j > 0:
            path.append((i, j))
            min_cost_dir = np.argmin([AD[i-1, j-1], AD[i-1, j], AD[i, j-1]])
            if min_cost_dir == 0:
                i -= 1
                j -= 1
            elif min_cost_dir == 1:
                i -= 1
            else:
                j -= 1
        path.append((0, 0))
        path = path[::-1]  # Reverse the path to start from the beginning

        # Plotting
        plt.figure(figsize=(10, 8))
        plt.imshow(local_distances, cmap='viridis', origin='lower')
        plt.colorbar(label='Local Distance')
        # Unzip the path into two lists of x and y indices
        y, x = zip(*path)
        plt.plot(x, y, marker='o', color='r', markersize=3, linestyle='-', linewidth=1, label='Optimal Path')
        plt.xlabel('Sequence 2')
        plt.ylabel('Sequence 1')
        plt.title('DTW Optimal Path')
        plt.legend()
        plt.show()

    return d
