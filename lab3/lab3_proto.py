import numpy as np
from lab3_tools import *
from lab2_proto import *
from lab1_proto import mfcc, mspec

def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
   """ word2phones: converts word level to phone level transcription adding silence

   Args:
      wordList: list of word symbols
      pronDict: pronunciation dictionary. The keys correspond to words in wordList
      addSilence: if True, add initial and final silence
      addShortPause: if True, add short pause model "sp" at end of each word
   Output:
      list of phone symbols
   """
   # We iterate over every word and add their phonemes to a list
   phone_list = []
   for word in wordList:
      for phoneme in pronDict[word]: phone_list.append(phoneme)
      
      # If we have pauses between words, we add a pause token
      if addShortPause: phone_list.append("sp")
   
   # If we have silence, we add a silence token at the start and end
   if addSilence: phone_list = ["sil"] + phone_list + ["sil"]      
   return phone_list

def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
   """ forcedAlignmen: aligns a phonetic transcription at the state level

   Args:
      lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
            computed the same way as for the training of phoneHMMs
      phoneHMMs: set of phonetic Gaussian HMM models
      phoneTrans: list of phonetic symbols to be aligned including initial and
                  final silence

   Returns:
      list of strings in the form phoneme_index specifying, for each time step
      the state from phoneHMMs corresponding to the viterbi path.
   """
   utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)

   phones = sorted(phoneHMMs.keys())
   nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
   stateTrans = [phone + '_' + str(stateID) for phone in phoneTrans for stateID in range(nstates[phone])]

   obslogik = log_multivariate_normal_density_diag(lmfcc, utteranceHMM["means"], utteranceHMM["covars"])  
   _, viterbi_path = viterbi(obslogik, np.log(utteranceHMM["log_startprob"][:-1]), np.log(utteranceHMM["log_transmat"][:-1, :-1]), forceFinalState=True)

   # Convert state in utteranceHMM to unique state id in stateList
   # Ex. state 10 becomes "r_1"
   viterbiStateIDPath = [stateTrans[i] for i in viterbi_path]

   return viterbiStateIDPath

def extractData(filepath, phoneHMMs, stateList, prondict):
   data = []
   for root, dirs, files in os.walk(filepath):
      for file in files:
         if file.endswith('.wav'):
            # Reading samples and samling rate from current file utterance
            filename = os.pot.join(root, file)
            samples, samplingrate = loadAudio(filename)

            # Functions from lab 1 to read lmfcc and mspec
            lmfcc = mfcc(samples)
            mspec = mspec(samples)
            
            # Computing the word and phoneme transition matrix for current utterance
            wordTrans = list(path2info(filename)[2])
            phoneTrans = words2phones(wordTrans, prondict)

            # Current utterance sequence might be like 'ah_0', 'ah_1'
            target_phones = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)
            # phoneme id sequence would then be: 0, 1
            target_phones_id = [stateList[phone] for phone in target_phones]

            data.append({'filename': filename, 'lmfcc': lmfcc, 
                           'mspec': mspec, 'targets': target_phones_id})
   return data