import numpy as np 
import  matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from lab1_tools import *
from lab1_proto import *

def compute_mfcc_features(data):
    mfcc_features = []

    # Iterate over all utterances in the data set
    for utterance in data:
        # Compute MFCC features for current utterance and add to feature array
        utterance_mfcc = mfcc(utterance["samples"]) # Return (N x 13) array, where N is the window amount for this utterance
        mfcc_features.append(utterance_mfcc)        # Add this (N x 13) array to our mfcc feature array

    # We have a (K x N x 13) array, which we concatenate so it gets the shape (K*N, 13)
    mfcc_feature_arr = np.concatenate(mfcc_features, axis=0)

    print("MFCC Feature Array shape:", mfcc_feature_arr.shape)
    return mfcc_feature_arr

def compute_mspec_features(data):
    mspec_features = []

    # Iterate over all utterances in the data set
    for utterance in data:
        # Compute MFCC features for current utterance and add to feature array
        utterance_mspec = mspec(utterance["samples"]) # Return (N x 13) array, where N is the window amount for this utterance
        mspec_features.append(utterance_mspec)        # Add this (N x 13) array to our mfcc feature array

    # We have a (K x N x 13) array, which we concatenate so it gets the shape (K*N, 13)
    mspec_feature_arr = np.concatenate(mspec_features, axis=0)

    print("MSPEC Feature Array shape:", mspec_feature_arr.shape)
    return mspec_feature_arr

def compute_correlation_matrix(feature_arr):
    # Compute the correlation matrix
    correlation_matrix = np.corrcoef(feature_arr, rowvar=False) # rowvar is false as rows = observations and cols = coefficients
    return correlation_matrix


def gmm_visualize(gm, mffc_samples):
  posteriors = []
  for sample in mffc_samples:
    posteriors.append(gm.predict_proba(sample))

  plt.figure(figsize=(12, 8))

  for i, posterior in enumerate(posteriors):
    plt.subplot(2, 2, i+1)
    plt.pcolormesh(posterior.T, cmap='viridis')
    plt.title(f'GMM Posteriors {i}')
    plt.colorbar()

  plt.tight_layout()
  plt.show()

def local_dist(utterance_1, utterance_2):
    mffc_1 = mfcc(utterance_1)
    mffc_2 = mfcc(utterance_2)

    # Compute the Euclidean distance between each pair
    dist_arr = cdist(mffc_1, mffc_2)
    return dist_arr