from helpers import get_logger
import regex as re
import numpy as np
import nltk
import logging
from pathlib import Path

import pandas as pd
import string
from collections import Counter

import pickle
import tqdm
from datasets import load_dataset
#import stopwords
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def check_pos_tag(triplet, allowed_pos_tags_subject, allowed_pos_tags_object):
    """ Check if the pos tags of the subject and object are in the allowed pos tags

    Parameters:
        triplet (tuple): A triplet (subject, verb, object)
        allowed_pos_tags_subject (list[str]): A list of allowed pos tags for the subject
        allowed_pos_tags_object (list[str]): A list of allowed pos tags for the object

    Returns:
        tuple: A tuple containing a boolean indicating if the pos tags are allowed, and a string indicating if the subject or object is not allowed

    """
    subject, verb, object = triplet
    # use pos tagging on the subject, check if the pos tag is in allowed_pos_tags
    subject_pos_tags = nltk.pos_tag(nltk.word_tokenize(subject))
    object_pos_tags = nltk.pos_tag(nltk.word_tokenize(object))
    # if there is no allowed pos tag in any of the subject words
    if all([pos_tag not in allowed_pos_tags_subject for _, pos_tag in subject_pos_tags]):
        return False, 'subject'
    if all([pos_tag not in allowed_pos_tags_object for _, pos_tag in object_pos_tags]):
        return False, 'object'
    return True, None

def filter_pos_tag_triplet(subject_pos_tags, object_pos_tags, subject, verb, object, logger, allowed_pos_tags_subject, allowed_pos_tags_object):
    """ Filter the pos tags of the subject and object

    Parameters:
        subject_pos_tags (list[tuple]): A list of tuples containing the pos tags of the subject
        object_pos_tags (list[tuple]): A list of tuples containing the pos tags of the object
        subject (str): The subject
        verb (str): The verb
        object (str): The object
        logger (logging.Logger): The logger
        allowed_pos_tags_subject (list[str]): A list of allowed pos tags for the subject
        allowed_pos_tags_object (list[str]): A list of allowed pos tags for the object

    Returns:
        tuple: A tuple containing the new subject, verb and object
    """

    # maintain the pos tags in the subject and object, if the subject or object changes, write to the log file
    new_subject = ' '.join([word for word, pos_tag in subject_pos_tags if pos_tag in allowed_pos_tags_subject])
    new_object = ' '.join([word for word, pos_tag in object_pos_tags if pos_tag in allowed_pos_tags_object])
    if new_subject != subject:
        logger.info('Old subject: ' + subject + '\n' + 'New subject: ' + new_subject + '\n' + '\n')
    if new_object != object:
        logger.info('Old object: ' + object + '\n' + 'New object: ' + new_object + '\n' + '\n')
    return (new_subject, verb, new_object)

def filter_pos_tag(df, logger, allowed_pos_tags_subject=['NN', 'NNS', 'NNP', 'NNPS'], allowed_pos_tags_object=['NN', 'NNS', 'NNP', 'NNPS', 'VBG', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
    """ Filter the pos tags of the subject and object

    Parameters:
        df (pd.DataFrame): The dataframe containing the triplets
        logger (logging.Logger): The logger
        allowed_pos_tags_subject (list[str]): A list of allowed pos tags for the subject
        allowed_pos_tags_object (list[str]): A list of allowed pos tags for the object

    Returns:
        pd.DataFrame: The dataframe containing the filtered triplets
    """

    # write allowed pos tags to the log file
    logger.info('Allowed POS tags for the subject: ' + str(allowed_pos_tags_subject))
    logger.info('Allowed POS tags for the object: ' + str(allowed_pos_tags_object))
    logger.info('Filtering the triplets based on the POS tags of the subject and object')
    for index, row in df.iterrows():
        new_triplets = []
        for triplet in row['triplets']:
            subject, verb, object = triplet
            subject_pos_tags = nltk.pos_tag(nltk.word_tokenize(subject))
            object_pos_tags = nltk.pos_tag(nltk.word_tokenize(object))
            # if the subject does not have any of the allowed pos tags, remove the triplet
            if all([pos_tag not in allowed_pos_tags_subject for _, pos_tag in subject_pos_tags]):
                # write to the log file
                logger.info('Removed triplet ' + str(triplet) + ' because the subject does not have any of the allowed POS tags.')
                continue
            # if the object does not have any of the allowed pos tags, remove the triplet
            if all([pos_tag not in allowed_pos_tags_object for _, pos_tag in object_pos_tags]):
                # write to the log file
                logger.info('Removed triplet ' + str(triplet) + ' because the object does not have any of the allowed POS tags.')
                continue
            # keep only the pos tags that are in the allowed_pos_tags
            new_triplets.append(filter_pos_tag_triplet(subject_pos_tags, object_pos_tags, subject, verb, object, logger, allowed_pos_tags_subject, allowed_pos_tags_object))
        df.at[index, 'triplets'] = new_triplets
    return df

def lower_case(df):
    """ Lower case the triplets

    Parameters:
        df (pd.DataFrame): The dataframe containing the triplets

    Returns:
        pd.DataFrame: The dataframe containing the lower cased triplets
    """
    df['triplets'] = df['triplets'].apply(lambda x: [(subject.lower(), verb.lower(), object.lower()) for subject, verb, object in x])
    return df

def filter_length(df, logger, cutoff_length=10):
    """ Filter the triplets based on length

    Parameters:
        df (pd.DataFrame): The dataframe containing the triplets
        logger (logging.Logger): The logger
        cutoff_length (int): The cutoff length

    Returns:
        pd.DataFrame: The dataframe containing the filtered triplets
    """

    num_removed = 0
    for index, row in df.iterrows():
        new_triplets = []
        for i in range(len(row['triplets'])):
            triplet = row['triplets'][i]
            subject, verb, object = triplet
            if len(subject.split()) > cutoff_length or len(verb.split()) > cutoff_length or len(object.split()) > cutoff_length:
                logger.info('Removed triplet ' + str(triplet) + ' because the length of the subject, verb or object exceeds the cutoff length.')
                num_removed += 1
            else:
                new_triplets.append(triplet)
        df.at[index, 'triplets'] = new_triplets
    logger.info('Removed ' + str(num_removed) + ' triplets in total.')
    return df

def lemmatize(df, logger):
    """ Lemmatize the triplets

    Parameters:
        df (pd.DataFrame): The dataframe containing the triplets
        logger (logging.Logger): The logger

    Returns:
        pd.DataFrame: The dataframe containing the lemmatized triplets
    """

    logger.info('Lemmatizing the triplets')
    lemmatizer = WordNetLemmatizer()
    triplets_lemmatized = []
    triplets = df['triplets'].values
    for triplet_list in triplets:
        new_triplet_list = []
        for triplet in triplet_list:
            subject, verb, object = triplet
            # subject, verbs and objects are lemmatized, note that they may consist of multiple words
            subject_new = ' '.join([lemmatizer.lemmatize(word) for word in subject.split()])
            object_new = ' '.join([lemmatizer.lemmatize(word) for word in object.split()])
            verb_new = ' '.join([lemmatizer.lemmatize(word) for word in verb.split()])
            # If the subject, verb or object changed, log it
            if subject_new != subject:
                logger.info(f"Subject: {subject} -> {subject_new}")
            if object_new != object:
                logger.info(f"Object: {object} -> {object_new}")
            if verb_new != verb:
                logger.info(f"Verb: {verb} -> {verb_new}")
            new_triplet_list.append((subject_new, verb_new, object_new))
        # append the new triplet list
        triplets_lemmatized.append(new_triplet_list)
    # update the dataframe
    df['triplets'] = triplets_lemmatized
    return df

def keep_only_text(df, logger):
    """ Keep only the text in the triplets

    Parameters:
        df (pd.DataFrame): The dataframe containing the triplets
        logger (logging.Logger): The logger

    Returns:
        pd.DataFrame: The dataframe containing the triplets with only text
    """
    logger.info('Removing punctuation')
    # we want to keep letters, numbers and hyphens, but remove any other character
    to_keep = string.ascii_letters + string.digits + "-" + " "
    triplets = df['triplets'].values
    triplets_no_punctuation = []
    for triplet_list in triplets:
        new_triplet_list = []
        for triplet in triplet_list:
            subject, verb, object = triplet
            # keep only the characters that are in to_keep
            subject_new = ''.join([c for c in subject if c in to_keep])
            object_new = ''.join([c for c in object if c in to_keep])
            verb_new = ''.join([c for c in verb if c in to_keep])
            # if the subject, verb or object changed, log it
            if subject_new != subject:
                logger.info(f"Subject: {subject} -> {subject_new}")
            if object_new != object:
                logger.info(f"Object: {object} -> {object_new}")
            if verb_new != verb:
                logger.info(f"Verb: {verb} -> {verb_new}")
            new_triplet_list.append((subject_new, verb_new, object_new))
        # append the new triplet list
        triplets_no_punctuation.append(new_triplet_list)
    # update the dataframe
    df['triplets'] = triplets_no_punctuation
    return df

def remove_stopwords(df, logger, redundant_verbs=['can', 'will', 'shall', 'may', 'could', 'would', 'should', 'has', 'are', 'is', 'have', 'was', 'were', 'had', 'do', 'does', 'did', 'am', 'be', 'being', 'been', 'get', 'got', 'gets', 'getting', 'gotten', 'make', 'makes', 'made', 'making', 'let', 'lets', 'letting', 'let']):
    """ Remove the stopwords from the triplets

    Parameters:
        df (pd.DataFrame): The dataframe containing the triplets
        logger (logging.Logger): The logger
        redundant_verbs (list[str]): A list of redundant verbs

    Returns:
        pd.DataFrame: The dataframe containing the triplets with the stopwords removed
    """

    logger.info('Removing stopwords')
    for idx, row in df.iterrows():
        triplets_col = row['triplets']
        new_triplets = []
        for triplet in triplets_col:
            subject, verb, object = triplet
            # if verb has multiple words, remove the redundant ones
            verbs = verb.split()
            if len(verbs) > 1:
                verbs = [v for v in verbs if v not in redundant_verbs]
                verb = ' '.join(verbs)
            # Remove stopwords from subject, verb and object, if any of them is empty, do not append the triplet
            new_subject = ' '.join([word for word in subject.split() if word not in stopwords.words('english')])
            new_object = ' '.join([word for word in object.split() if word not in stopwords.words('english')])
            if len(new_subject) > 0 and len(new_object) > 0:
                new_triplets.append((new_subject, verb, new_object))
            if new_subject != subject:
                logger.info(f"Removed stopwords from subject: {subject} -> {new_subject}")
            if new_object != object:
                logger.info(f"Removed stopwords from object: {object} -> {new_object}")
        df.at[idx, 'triplets'] = new_triplets
    return df

def clean_up_triplets(df, logger):
    """ Clean up the triplets

    Parameters:
        df (pd.DataFrame): The dataframe containing the triplets
        logger (logging.Logger): The logger

    Returns:
        pd.DataFrame: The dataframe containing the cleaned up triplets
    """
    logger.info('Cleaning up the triplets')
    # Remove any row that has no triplets at all
    df = df[df['triplets'].apply(lambda x: len(x) > 0)]
    # remove triplets that are empty
    num_removed = 0
    for idx, row in df.iterrows():
        triplets_col = row['triplets']
        new_triplets = []
        for triplet in triplets_col:
            subject, verb, object = triplet
            subject_words = subject.split()
            # Remove any part of the subject that has less than 3 characters
            subject = ' '.join([word for word in subject_words if len(word) > 2])
            object_words = object.split()
            # Remove any part of the object that has less than 2 characters
            object = ' '.join([word for word in object_words if len(word) > 2])
            # if the subject or object has less than 3 characters, remove the triplet
            if len(subject) < 3 or len(object) < 3 or len(verb) < 1:
                logger.info(f"Removed triplet {triplet} because the subject or object has less than 3 characters, or the verb is empty.")   
                num_removed += 1
                continue
            # Replace multiple subsequent spaces with a single space
            subject = re.sub(' +', ' ', subject)
            object = re.sub(' +', ' ', object)
            verb = re.sub(' +', ' ', verb)
            new_triplets.append((subject, verb, object))
        df.at[idx, 'triplets'] = new_triplets
    return df

def get_words_from_triplets(df):
    """ Get the words from the triplets

    Parameters:
        df (pd.DataFrame): The dataframe containing the triplets

    Returns:
        list[str]: A list of words
    """
    triplets = df['triplets'].tolist()
    # flatten the list
    triplets = [item for sublist in triplets for item in sublist]
    # triplets is a list of tuples
    triplets = [list(t) for t in triplets]
    # flatten the list
    triplets = [item for sublist in triplets for item in sublist]
    # There may be multiple words, split based on space
    triplets = [t.split() for t in triplets]
    # flatten the list
    triplets = [item for sublist in triplets for item in sublist]
    # Get unique words
    words = list(set(triplets))
    # remove words that are 1 character long
    words = [w for w in words if len(w)>1]
    return words

def filter_book_corpus(book_corpus, max_length):
    """ Filter the book corpus based on length

    Parameters:
        book_corpus (list[str]): The book corpus
        max_length (int): The maximum length

    Returns:
        list[str]: The filtered book corpus
    """
    # remove all books with length over max_length
    book_corpus = [book for book in book_corpus if len(book) < max_length]
    return book_corpus

def get_corpus_frequency(corpus, words, subset=1, min_terms = 5):
    """ Get the frequency of the words in the corpus

    Parameters:
        corpus (list[str]): The corpus
        words (list[str]): The words
        subset (float): The fraction of the corpus to use
        min_terms (int): The minimum number of terms

    Returns:
        Counter: A counter containing the frequency of the words
    """

    # get fraction subset of the corpus
    corpus = corpus[:int(len(corpus)*subset)]
    document_counts = Counter()
    for i in tqdm.tqdm(range(len(corpus))):
        words_book = corpus[i].split()
        # lower case the words
        words_book = [word.lower() for word in words_book]
        # Only consider the words that are present at least min_terms times
        for word in words:
            if words_book.count(word) >= min_terms:
                document_counts[word] += 1
    return document_counts

def get_frequency_papers(PATH_FOLDER, words, min_terms = 5):
    """ Get the frequency of the words in the papers

    Parameters:
        PATH_FOLDER (Path): The folder containing the papers
        words (list[str]): The words
        min_terms (int): The minimum number of terms

    Returns:
        Counter: A counter containing the frequency of the words
    """
    # get all the files in the folder
    files = PATH_FOLDER.glob('*.txt')
    word_freq = Counter()
    for file in tqdm.tqdm(files):
        with open(PATH_FOLDER.joinpath(file), 'r', encoding='utf-8') as f:
            text = f.read()
            words_paper = text.split()
            # lower case the words
            words_paper = [word.lower() for word in words_paper]
            # Only consider the words that are present at least min_terms times
            for word in words:
                if words_paper.count(word) >= min_terms:
                    word_freq[word] += 1
    return word_freq

def compute_term_scores(corpus_freq, paper_freq, num_docs_corpus, num_docs_paper, min_paper_count=10):
    """ Compute the term scores

    Parameters:
        corpus_freq (Counter): The frequency of the words in the corpus
        paper_freq (Counter): The frequency of the words in the papers
        num_docs_corpus (int): The number of documents in the corpus
        num_docs_paper (int): The number of documents in the papers
        min_paper_count (int): The minimum number of papers

    Returns:
        dict: A dictionary containing the term scores
    """
    # Now make a new dictionary that is the multiplication of the two. If a key is in corpus_freq but not in paper_freq, we set the value to 0
    # if a key is in paper_freq but not in corpus_freq, and the value in paper_freq is at least 10, we set the value to infinity.
    # if a key is in paper_freq but not in corpus_freq, and the value in paper_freq is less than 10, we set the value to 0
    term_scores = {}
    for key in corpus_freq:
        if key in paper_freq and paper_freq[key] >= min_paper_count:
            corpus_score = - np.log(corpus_freq[key]/num_docs_corpus)
            paper_score = np.log(paper_freq[key]/num_docs_paper)
            term_scores[key] = corpus_score + paper_score
        else:
            term_scores[key] = - np.inf
    for key in paper_freq:
        if key not in corpus_freq:
            if paper_freq[key] >= min_paper_count:
                term_scores[key] = np.inf
            else:
                term_scores[key] = - np.inf
    # From the dictionary, remove all keys where there is punctuation, and remove all keys existing of only numbers
    term_scores_filtered = {key: value for key, value in term_scores.items() if (not any(char in string.punctuation for char in key) and not key.isdigit() and not len(key) < 2)}
    # sort term_scores from highest score to lowest
    term_scores_filtered = dict(sorted(term_scores_filtered.items(), key=lambda x: x[1], reverse=True))
    return term_scores_filtered

def filter_triplets(triplets, term_scores, threshold = 0.1):
    """ Filter the triplets based on the term scores

    Parameters:
        triplets (pd.DataFrame): The dataframe containing the triplets
        term_scores (dict): A dictionary containing the term scores
        threshold (float): The threshold

    Returns:
        pd.DataFrame: The dataframe containing the filtered triplets
        list[tuple]: A list of removed triplets
    """

    # Retain the triplets that have at least one word with a score in the top threshold percent
    removed_triplets = []
    filtered_triplets = triplets.copy()

    # As a threshold, take the score of the word at the threshold percentile
    threshold_score = list(term_scores.values())[int(len(term_scores) * threshold)]
    for idx, row in tqdm.tqdm(triplets.iterrows()):
        triplet_list = row['triplets']
        kept_triplets = []
        for triplet in triplet_list:
            subject, verb, object = triplet
            object_words = object.split()
            verb_words = verb.split()
            subject_words = subject.split()
            # check if any of the words is not in the dictionary, if so, put it at negative infinity
            all_words = object_words + verb_words + subject_words
            for word in all_words:
                if word not in term_scores:
                    term_scores[word] = - np.inf
            if any([term_scores[word] >= threshold_score for word in subject_words]):
                kept_triplets.append(triplet)
            else:
                removed_triplets.append(triplet)
        filtered_triplets.at[idx, 'triplets'] = kept_triplets
    return filtered_triplets, removed_triplets

def filter_with_bookcorpus(triplets, logger, path_book_corpus, path_papers, path_book_freq, path_paper_freq, max_length_book=30000, threshold=0.1, min_paper_count=10):
    """ Filter the triplets with the book corpus

    Parameters:
        triplets (pd.DataFrame): The dataframe containing the triplets
        path_book_corpus (str): The path to the book corpus
        path_papers (str): The path to the papers
        path_book_freq (str): The path to the book frequency
        path_paper_freq (str): The path to the paper frequency
        max_length_book (int): The maximum length of the book
        threshold (float): The threshold
        min_paper_count (int): The minimum number of papers that need to contain a word for it to be considered

    Returns:
        pd.DataFrame: The dataframe containing the filtered triplets
        list[tuple]: A list of removed triplets
    """

    # Load bookcorpus
    if path_book_corpus.exists():
        logger.info("Loading book corpus from file")
        book_corpus_gutenberg = pickle.load(open(path_book_corpus, "rb"))
    else:
        logger.info("Loading book corpus from huggingface")
        book_corpus_gutenberg = load_dataset("sedthh/gutenberg_english")['train']['TEXT']
        book_corpus_gutenberg = filter_book_corpus(book_corpus_gutenberg, max_length_book)
        # save book corpus
        pickle.dump(book_corpus_gutenberg, open(path_book_corpus, "wb"))
    words_in_triplets = get_words_from_triplets(triplets)


    #if it doesn't exist, create it
    if not path_book_freq.exists():
        logger.info("Creating book frequency")
        word_freq_book = get_corpus_frequency(book_corpus_gutenberg, words_in_triplets, subset=0.3, min_terms=5)
        with open(path_book_freq, 'wb') as f:
            pickle.dump(word_freq_book, f)
    else:
        logger.info("Loading book frequency")
        with open(path_book_freq, 'rb') as f:
            word_freq_book = pickle.load(f)
            
    if not path_paper_freq.exists():
        logger.info("Creating paper frequency")
        word_freq_papers = get_frequency_papers(path_papers, words_in_triplets, min_terms=5)
        with open(path_paper_freq, 'wb') as f:
            pickle.dump(word_freq_papers, f)
    else:
        logger.info("Loading paper frequency")
        with open(path_paper_freq, 'rb') as f:
            word_freq_papers = pickle.load(f)
            
    num_docs_corpus = len(book_corpus_gutenberg)
    num_docs_paper = len(list(path_papers.glob('*.txt')))
    term_scores = compute_term_scores(word_freq_book, word_freq_papers, num_docs_corpus, num_docs_paper, min_paper_count=min_paper_count)
    filtered_triplets, removed_triplets = filter_triplets(triplets, term_scores, threshold=threshold)
    return filtered_triplets, removed_triplets

def main():
    ###################################   SETTINGS  ###################################################
    CUTOFF_LENGTH = 6
    THRESHOLD_BOOKCORPUS = 0.1
    MIN_PAPER_COUNT = 10

    # PATHS
    PATH_ROOT = Path.cwd()

    ################################## FILL IN THE PATHS ###############################################
    PATH_PROCESSED_TEXT = PATH_ROOT.joinpath('') # path to the folder with the processed text
    PATH_TRIPLETS = PATH_ROOT.joinpath('') # path to the folder with the triplets
    PATH_LOG = PATH_ROOT.joinpath('') # path to the log file
    ####################################################################################################
    
    # if folders for claims and triplets dont exist, make them
    if not PATH_TRIPLETS.exists():
        PATH_TRIPLETS.mkdir()

    if not PATH_LOG.exists():
        PATH_LOG.mkdir()
    
    PATH_BOOK_CORPUS = PATH_ROOT.joinpath("/book_corpus_gutenberg.pkl")
    PATH_SAVE_BOOK_FREQ = PATH_TRIPLETS.joinpath("word_freq_book.pkl")
    PATH_SAVE_PAPER_FREQ = PATH_TRIPLETS.joinpath("word_freq_papers.pkl")

    PATH_LOG_GENERAL_OUTPUT = PATH_LOG.joinpath('general_output_process_triplets.txt')
    PATH_LOG_FILTER_POS = PATH_LOG.joinpath('log_filterpos.txt')
    PATH_LOG_LENGTH = PATH_LOG.joinpath('log_length.txt')
    PATH_LOG_MAINTAIN_POS = PATH_LOG.joinpath('log_maintainpos.txt')
    PATH_LOG_LEMMATIZE = PATH_LOG.joinpath('log_lemmatize.txt')
    PATH_LOG_KEEPTEXT = PATH_LOG.joinpath('log_keeptext.txt')
    PATH_LOG_CLEANUP = PATH_LOG.joinpath('log_cleanup.txt')
    PATH_LOG_STOPWORDS = PATH_LOG.joinpath('log_stopwords.txt')
  
    general_logger = get_logger('general_logger_process_triplets', PATH_LOG_GENERAL_OUTPUT)
    
    # Clear the log files
    open(PATH_LOG_FILTER_POS, 'w').close()
    open(PATH_LOG_LENGTH, 'w').close()
    open(PATH_LOG_MAINTAIN_POS, 'w').close()
    open(PATH_LOG_LEMMATIZE, "w").close()
    open(PATH_LOG_KEEPTEXT, "w").close()
    open(PATH_LOG_CLEANUP, "w").close()
    open(PATH_LOG_STOPWORDS, "w").close()

   
    df = pd.read_csv(PATH_TRIPLETS + 'triplets.csv')
    df['triplets'] = df['triplets'].apply(lambda x: eval(x))
    # lower case everything
    df = lower_case(df)
    # Remove triplets where the length of the subject or object exceeds the cutoff length
    df = filter_length(df, get_logger('log_length', PATH_LOG_LENGTH), cutoff_length=CUTOFF_LENGTH)
    # Remove stopwords from the triplets
    df = remove_stopwords(df, get_logger('log_stopwords', PATH_LOG_STOPWORDS))
    # Remove any character that is not text, except for hyphens
    df = keep_only_text(df, get_logger('log_keeptext', PATH_LOG_KEEPTEXT))
    # Remove triplets that do not have a subject or object with allowed pos tags, possible to specify the allowed pos tags
    df = filter_pos_tag(df, get_logger('log_filterpos', PATH_LOG_FILTER_POS))
    # Lemmatize the triplets
    df = lemmatize(df, get_logger('log_lemmatize', PATH_LOG_LEMMATIZE))
    # Clean up the triplets, remove triplets that are empty or have a subject or object with less than 3 characters
    df = clean_up_triplets(df, get_logger('log_cleanup', PATH_LOG_CLEANUP))
    # Filter the triplets with the book corpus
    df, removed_triplets = filter_with_bookcorpus(df, general_logger, PATH_BOOK_CORPUS, PATH_PROCESSED_TEXT, PATH_SAVE_BOOK_FREQ, PATH_SAVE_PAPER_FREQ, threshold=THRESHOLD_BOOKCORPUS, min_paper_count=MIN_PAPER_COUNT)
    # Save the triplets
    
    df.to_csv(PATH_TRIPLETS + 'processed_triplets.csv', index=False)
    # save removed triplets
    with open(PATH_TRIPLETS + 'removed_triplets.pkl', 'wb') as f:
        pickle.dump(removed_triplets, f)
    general_logger.info('Triplets are processed and saved in ' + PATH_TRIPLETS.joinpath('processed_triplets.csv'))

if __name__ == "__main__":
    main()



