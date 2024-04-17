import numpy as np
from lab2_tools import *

def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models
   
    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """

    HMM1_dim = hmm1["startprob"].shape[0]
    HMM2_dim = hmm2["startprob"].shape[0]
    N = HMM1_dim + HMM2_dim - 1

    # We combine the two matrices but eliminate the non-emiting state between the models
    # Ex, PI_1 = 1x4 and PI_2 = 1x4 --> PI_Concat = 1x7
    PI_concat = np.zeros(shape= (N))
    transmat_concat = np.zeros(shape=(N, N))

    for col_id in range(N):
      if col_id < HMM1_dim - 1:
        # Just HMM1 values
        PI_concat[col_id] = hmm1["startprob"][col_id]
      else:
        # Last startprob of HMM1  multiplied by startprob of HMM2 for current column
        PI_concat[col_id] = hmm1["startprob"][-1] * hmm2["startprob"][(col_id + 1) % HMM2_dim]

    for row_id in range(N-1):
      for col_id in range(N):
        if col_id < HMM1_dim - 1 and row_id < HMM1_dim - 1:
          # Just HMM1 values
          transmat_concat[row_id][col_id] = hmm1["transmat"][row_id][col_id]
        elif col_id > HMM1_dim - 1 and row_id < HMM1_dim:
          # last value of HMM1 multiplied by startprob of HMM2 for current column
          transmat_concat[row_id][col_id] = hmm1["transmat"][row_id][-1] * hmm2["startprob"][(col_id + 1) % HMM2_dim]
        else:
          # Just HMM2 values
          transmat_concat[row_id][col_id] = hmm2["transmat"][(row_id+1) % HMM2_dim][(col_id+1)% HMM2_dim]

    # The final row is filled with zeros except for in the last column where there's a one
    transmat_concat[-1][-1] = 1

    means_concat = np.concatenate((hmm1["means"], hmm2["means"]), axis=0)
    covars_concat = np.concatenate((hmm1["covars"], hmm2["covars"]), axis=0)

    concat_HMM = {"startprob": PI_concat, "transmat": transmat_concat, "means": means_concat, "covars": covars_concat}
    return concat_HMM

# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """

    # Convert weights to log space to use in log-sum-exp
    log_weights = np.log(weights)

    # Log-sum-exp trick to avoid numerical underflow:
    # For each observation, compute the log-sum-exp of the weighted log likelihoods across all components
    max_log_emlik = np.max(log_emlik, axis=1, keepdims=True)
    weighted_log_probs = log_emlik + log_weights
    logsumexp = max_log_emlik + np.log(np.sum(np.exp(weighted_log_probs - max_log_emlik), axis=1))

    # Sum log likelihoods over all observations
    gmmloglik = np.sum(logsumexp)

    return gmmloglik


def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    print(log_startprob.shape)
    print(log_transmat.shape)

    N, M = log_emlik.shape
    forward_prob = np.zeros(log_emlik.shape)

    forward_prob[0, :] = log_startprob[:-1] + log_emlik[0, :]

    for n in range(1, N):
        for m in range(M):
            forward_prob[n, m] = logsumexp(forward_prob[n-1, :] + log_transmat[:-1, m]) + log_emlik[n, m]

    return forward_prob


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    N, M = log_emlik.shape
    backward_prob = np.zeros_like(log_emlik)

    for n in range(N-2, -1, -1):
      for m in range(M):
        # transition prob from current state to all state + emission prob from prev frame + backward prob from prev frame
        backward_prob[n][m] = logsumexp( log_transmat[m][:-1] + log_emlik[n+1][:] + backward_prob[n+1][:] )

    return backward_prob


def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """

    N, M = log_emlik.shape
    V = np.full((N, M), -np.inf)  # Log probabilities, initialized very small
    B = np.zeros((N, M), dtype=int)  # Backtracking pointers

    # Initialization
    V[0, :] = log_startprob + log_emlik[0, :]

    # Dynamic programming
    for n in range(1, N):
        for j in range(M):
            # Vector of probabilities from each state i at time n-1 to state j at time n
            log_probs = V[n-1, :] + log_transmat[:, j] + log_emlik[n, j]
            V[n, j] = np.max(log_probs)
            B[n, j] = np.argmax(log_probs)

    # Termination: Find the best last state
    if forceFinalState:
        viterbi_loglik = V[-1, -1]  # force the path to end in the last state
        last_state = M - 1
    else:
        viterbi_loglik = np.max(V[-1, :])
        last_state = np.argmax(V[-1, :])

    # Backtrack to find the best path
    viterbi_path = np.zeros(N, dtype=int)
    viterbi_path[-1] = last_state
    for n in range(N-2, -1, -1):
        viterbi_path[n] = B[n+1, viterbi_path[n+1]]

    return viterbi_loglik, viterbi_path


def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    N, M = log_alpha.shape

    log_gamma = log_alpha + log_beta - logsumexp(log_alpha[N-1])
    return log_gamma


def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """

    gamma = np.exp(log_gamma)  # Convert log posteriors to probabilities
    sum_gamma = np.sum(gamma, axis=0)  # Sum of posteriors for each state

    # Ensure no zero divisions; relevant when no significant posterior weight is assigned to a state
    safe_sum_gamma = np.where(sum_gamma == 0, 1, sum_gamma)

    # Calculate means
    means = np.dot(gamma.T, X) / safe_sum_gamma[:, None]  # MxD

    # Calculate covariances
    covars = np.zeros_like(means)  # MxD
    for m in range(means.shape[0]):  # Iterate over each state
        # Weighted sum of squared differences
        diff = X - means[m]
        weighted_sumsq = np.dot(gamma[:, m] * diff.T, diff)
        covars[m] = weighted_sumsq / safe_sum_gamma[m]

    # Apply variance floor
    covars[covars < varianceFloor] = varianceFloor

    return means, covars
