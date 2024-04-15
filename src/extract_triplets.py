from helpers import get_logger

import spacy
import textacy
import os
import pandas as pd

from pathlib import Path

nlp = spacy.load("en_core_web_lg")

def extract_triplets(folder, save_folder):
    """ Extract the triplets from the text in the folder and save them to a file

    Parameters:
        folder (str): The folder containing the text files
        save_folder (str): The folder to save the triplets to
    """

    dict = {}
    for filename in os.listdir(folder):
        with open(folder + filename, 'r', encoding='utf-8') as f:
            claim = f.read()
        doc = nlp(claim)
        triplets = textacy.extract.subject_verb_object_triples(doc)
        triplets_list = []
        for triplet in triplets:
            subject, verb, object = triplet
            # subject, verb, object is a list of tokens. Make it a string of tokens
            subject = ' '.join([token.text for token in subject])
            verb = ' '.join([token.text for token in verb])
            object = ' '.join([token.text for token in object])
            triplets_list.append((subject, verb, object))
        dict[filename] = triplets_list
    # make dataframe with 2 columns: paper and triplets. The column triplets should contain a list of all triplets
    df = pd.DataFrame(list(dict.items()), columns=['paper', 'triplets'])
    # save the dataframe as a csv file
    df.to_csv(save_folder + 'triplets.csv', index=False)

def main():
    ###################################   SETTINGS  ###################################################
    PATH_ROOT = Path.cwd()

    ################################## FILL IN THE PATHS ###############################################
    PATH_CLAIMS = PATH_ROOT.joinpath('')
    PATH_TRIPLETS = PATH_ROOT.joinpath('')
    PATH_LOG = PATH_ROOT.joinpath('')
    ####################################################################################################
    general_logger = get_logger('general_logger_extract_triplets', PATH_LOG.joinpath('general_output_extract_triplets.txt'))
    # if folders for claims and triplets dont exist, make them
    if not PATH_TRIPLETS.exists():
        PATH_TRIPLETS.mkdir()

    # check if there is the file PATH_TRIPLETS + 'triplets.csv', if not, extract the triplets
    if not PATH_TRIPLETS.joinpath('triplets.csv').exists():
        general_logger.info('Extracting triplets')
        extract_triplets(PATH_CLAIMS, PATH_TRIPLETS)
    else:
        general_logger.info('Triplets already extracted')

    general_logger.info('Triplets are extarcted and saved in ' + PATH_TRIPLETS.joinpath('triplets.csv'))

if __name__ == "__main__":
    main()