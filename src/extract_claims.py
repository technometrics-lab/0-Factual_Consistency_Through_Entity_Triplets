from helpers import get_logger

import nltk
import numpy as np
import tensorflow as tf

from pathlib import Path

def get_sentences(text, min_length=5):
    """ Split the text into sentences and remove sentences under a certain length

    Parameters: 
        text (str): The text to split into sentences
        min_length (int): The minimum length of a sentence to keep

    Returns:
        list[str]: A list of sentences [sentence1, sentence2, ...]
    """
    text = text.replace('\n', ' ')
    sentences = nltk.sent_tokenize(text)
    sentences = [sentence for sentence in sentences if len(sentence.split()) > min_length]
    return sentences

def split_char(text):
  """ Split the text into characters

    Parameters:
        text (str): The text to split into characters

    Returns:
        str: A string of characters
  """
  return " ".join(list(text))

def preprocess_sentences(sentences):
    """ Preprocess the sentences to be used as input for the claim model

    Parameters:
        sentences (list[str]): A list of sentences

    Returns:
        tuple: A tuple containing two numpy arrays, the first containing the sentences, the second containing the characters of the sentences
    """
    chars = [split_char(sentence) for sentence in sentences]
    # Combine chars and tokens into a numpy array
    input_data = (np.array(sentences), np.array(chars))
    return input_data

def classify_sentences(text, claim_model, min_length_sentence=5, threshold=0.5):

    """ Classify the sentences in the text as claims or not claims

    Parameters:
        text (str): The text to classify
        claim_model (tf.keras.Model): The claim model
        min_length_sentence (int): The minimum length of a sentence to keep
        threshold (float): The threshold for classifying a sentence as a claim

    Returns:
        tuple: A tuple containing two numpy arrays, the first containing the predictions, the second containing the sentences
    """

    sentences = get_sentences(text, min_length_sentence)
    input_data = preprocess_sentences(sentences)
    # Get predictions from model
    preds = claim_model.predict(input_data)
    # preds have two values, along dimension 1. The first value is the probability of the sentence being a claim, the second is the probability of it not being a claim.
    # If first value is greater than threshold, classify as claim, else classify as not claim
    preds = np.where(preds[:, 1] > threshold, 1, 0)
    
    return preds, sentences

def extract_claims(folder, save_folder, claim_model, logger, threshold=0.5):

    """ Extract the claims from the text in the folder and save them to a file

    Parameters:
        folder (str): The folder containing the text files
        save_folder (str): The folder to save the claims to
        claim_model (tf.keras.Model): The claim model
        log_errors (str): The file to write errors to
        threshold (float): The threshold for classifying a sentence as a claim
    """

    for file in folder.rglob('*.txt'):
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
        # classify the sentences
        if text == '':
            continue
        # try to classify the sentences, if it throws an error, write the error to a log file
        try:
            preds, sentences = classify_sentences(text, claim_model, threshold=threshold)
        except Exception as e:
            message = 'Error in file ' + file + ': ' + str(e) + '\n'
            logger.info(message)
        save_path = save_folder.joinpath(file.name)
        with open(save_path, 'w', encoding='utf-8') as f:
            for sentence, pred in zip(sentences, preds):
                if pred == 1:
                    f.write(sentence + '\n')


def main():
    ###################################   SETTINGS  ###################################################
    THRESHOLD_CLAIMS = 0.05
    PATH_ROOT = Path.cwd()

    ################################## FILL IN THE PATHS ###############################################
    PATH_PROCESSED_TEXT = PATH_ROOT.joinpath('') # path to the folder with the processed text
    PATH_CLAIMS = PATH_ROOT.joinpath('') # path to the folder where the claims should be saved
    PATH_MODEL = PATH_ROOT.joinpath("models/claim_model") # claimdistiller model should be here
    PATH_LOG = PATH_ROOT.joinpath('') # path to the log file
    ####################################################################################################
    # if folders for claims and triplets dont exist, make them
    if not PATH_CLAIMS.exists():
        PATH_CLAIMS.mkdir()

    if not PATH_LOG.exists():
        PATH_LOG.mkdir()

    general_logger = get_logger('general_logger_extract_claims', PATH_LOG.joinpath('general_output_extract_claims.txt'))

    PATH_LOG_ERROR_CLAIMS = PATH_LOG.joinpath('log_error_claims.txt')
    error_claim_logger = get_logger('log_error_claims', PATH_LOG_ERROR_CLAIMS)

    # Load the claim model
    claim_model = tf.keras.models.load_model(PATH_MODEL)

    #if there are not yet claims in the folder, extract . Use pathlib
    if len(list(PATH_CLAIMS.glob('*.txt'))) == 0:
        # write something to the file
        general_logger.info('Extracting claims')
        extract_claims(PATH_PROCESSED_TEXT, PATH_CLAIMS, claim_model=claim_model, logger=error_claim_logger, threshold=THRESHOLD_CLAIMS)
        general_logger.info('Claims are extracted and saved in ' + PATH_CLAIMS)

    else:
        general_logger.info('Claims are already extracted')

if __name__ == "__main__":
    main()