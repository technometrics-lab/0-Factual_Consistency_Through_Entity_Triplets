from helpers import get_logger, write_log
from pathlib import Path

from fastcoref import spacy_component
from abbreviations import schwartz_hearst

import re
import time
import tqdm
import spacy
from spacy.tokens import Doc

spacy.prefer_gpu() 

# Load spacy pipeline
nlp = spacy.load('en_core_web_lg', exclude=['parser', 'ner', 'lemmatizer', 'textcat'])
nlp.add_pipe("fastcoref")


def get_text_files_in_dir(path):
    """ Get all the .txt files in a directory
    
    Parameters:
        path (Path): path to the directory
        
        Return:
        texts (list[str]): list of texts
        text_file_names (list[str]): list of names of the text files  
    """
    texts = []
    text_file_names = []

    #iterate over txt files with Path.glob()
    for file in path.rglob('*.txt'):
        text_file_names.append(file.name)
        with open(file, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return texts, text_file_names

def remove_citations(texts, logger_citations):
    """Remove citations through a rule based heuristic

    Parameters:
        texts (list[str]): list of texts
        log_file (str): path to the log file
    Return:
        new_texts (list[str]): list of texts with the citations removed

    """
    new_texts = []
    for text in texts:
        new_text = ''
        line_count = 0
        for line in text.split('\n'):
            # Brackets with only a number inside are removed
            # Brackets with a year inside are removed
            # Brackets with a number inside and other text, e.g. [llm2], are not removed
            re_expression = '\[[0-9]{4}[a-zA-Z0-9 .,!/\-"\']*\]|\[[0-9]+\]|\[[a-zA-Z0-9 .,!/\-"\']*[0-9]{4}\]|\([a-zA-Z0-9 .,!/\-"\']*[0-9]{4}\)|\([0-9]{4}[a-zA-Z0-9 .,!/\-"\']*\)|\([0-9]+\)'
            if re.search(re_expression, line):
                # get starting and ending position of citation. If there are multiple citations in one line, store starting and ending position of each in a list
                new_line = re.sub(re_expression, '', line)
                start_pos, end_pos = [], []
                for match in re.finditer(re_expression, line):
                    start_pos.append(match.start())
                    end_pos.append(match.end())
                
                write_log(line, new_line, line_count, start_pos, end_pos, 'Removing citations', logger_citations)
            else:
                new_line = line
            line_count += 1
            new_text += new_line + '\n'
        new_texts.append(new_text)
    return new_texts

def expand_abbreviations(texts, logger_abbr):
    """Expand the abbreviations using the Schwartz-Hearst algorithm

    Parameters:
        texts (list[str]): list of texts
        log_file (str): path to the log file

    Return:
        new_texts (list[str]): list of texts with the abbreviations expanded
        pairs (dict): dictionary with the abbreviations as keys and the definitions as values
    """

    new_texts = []
    errors_with_abbreviations = set()
    for text in texts:
        pairs = schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=text)
        # Add the fully lowercased versions of the abbreviations as keys
        pairs_copy = pairs.copy()
        for abbrev, definition in pairs_copy.items():
            if abbrev.lower() != abbrev:
                pairs[abbrev.lower()] = definition
        # iterate over the lines in the text file and replace the abbreviations
        # split by \n to get the lines
  
        sentences = text.split('\n')
        new_sentences = []
        for i, sentence in enumerate(sentences):
            old_sentence = sentence
            start_pos, end_pos = [], []
            replacements = []
            for abbrev, definition in pairs.items():
                # check whether the abbreviation is in the sentence
                if abbrev in sentence:
                    # we have to make sure that the abbreviation is not inside a word, e.g. "in" in "within". It is allowed to have punctuation before and after the abbreviation, e.g. AI, or AI.
                    # We add a "try" since the abbreviation might contain a backslash, which would cause an error. If there is an error, we skip the abbreviation
                    try:
                        for m in re.finditer(abbrev, old_sentence):
                            # check whether there is a letter before and after the abbreviation
                            if m.start() > 0:
                                if sentence[m.start()-1].isalpha():
                                    continue
                            if m.end() < len(sentence):
                                if sentence[m.end()].isalpha():
                                    continue
                            replacements.append(((m.start(), m.end()), definition))
                    except:
                        errors_with_abbreviations.add(abbrev)
                        continue
            # Now we want to make sure that the replacements do not overlap. We do this by sorting the replacements by their start index and then iterating over them and only keeping the first replacement that does not overlap with the previous replacements
            replacements = sorted(replacements, key=lambda x: x[0][0])
            replacements_to_keep = []
            for replacement in replacements:
                if len(replacements_to_keep) == 0:
                    replacements_to_keep.append(replacement)
                else:
                    # check whether the replacement overlaps with the previous replacements
                    overlap = False
                    for replacement_to_keep in replacements_to_keep:
                        if replacement[0][0] <= replacement_to_keep[0][1]:
                            overlap = True
                            break
                    if not overlap:
                        replacements_to_keep.append(replacement)
            # Now we can replace the abbreviations with their definitions
            sorted_replacements_to_keep = sorted(replacements_to_keep, key=lambda x: x[0][0], reverse=True)
            for replacement in sorted_replacements_to_keep:
                sentence = sentence[:replacement[0][0]] + replacement[1] + sentence[replacement[0][1]:]
                start_pos.append(replacement[0][0])
                end_pos.append(replacement[0][1])
            new_sentences.append(sentence)
            if (len(replacements_to_keep) > 0):
                write_log(old_sentence, sentence, i, start_pos, end_pos, 'Abbreviation replacement', logger_abbr)
        # Get new_text by joining the sentences
        new_text = '\n'.join(new_sentences)
        new_texts.append(new_text)
    return new_texts, errors_with_abbreviations

def get_span_noun_indices(doc, cluster):
    """Get the indices of the spans that contain a noun

    Parameters:
        doc (Doc): spacy document
        cluster (list[tuple]): list of tuples with the start and end position of the spans

    Return:
        span_noun_indices (list[int]): list of indices of the spans that contain a noun
    """

    spans = [doc.text[span[0]:span[1]+1] for span in cluster]
    # We now want to know which tokens are in the spans and whether they are nouns
    span_noun_indices = []
    for idx, span in enumerate(spans):
        has_noun = False
        for token in doc:
            if token.text in span and token.pos_ in ['NOUN', 'PROPN']:
                has_noun = True
                break
        if has_noun:
            span_noun_indices.append(idx)
    return span_noun_indices

def is_containing_other_spans(span, all_spans):
    """Check whether a span is containing other spans

    Parameters:
        span (tuple): tuple with the start and end position of the span
        all_spans (list[tuple]): list of tuples with the start and end position of the spans

    Return:
        bool: whether the span is containing other spans
    """
    return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])

def get_cluster_head(doc: Doc, cluster, noun_indices):
    """Get the head of the cluster

    Parameters:
        doc (Doc): spacy document
        cluster (list[tuple]): list of tuples with the start and end position of the spans
        noun_indices (list[int]): list of indices of the spans that contain a noun

    Return:
        head_span (str): head of the cluster
        head_start_end (tuple): tuple with the start and end position of the head
    """
    head_idx = noun_indices[0]
    head_start, head_end = cluster[head_idx]
    head_span = doc.text[head_start:head_end+1]
    return head_span, (head_start, head_end)

def replace_corefs(doc, logger_coref, clusters):
    """Replace the coreferences in the text

    Parameters:
        doc (Doc): spacy document
        PATH_LOG (str): path to the log file
        clusters (list[list[tuple]]): list of clusters, where each cluster is a list of tuples with the start and end position of the spans

    Return:
        new_text (str): text with the coreferences replaced
    """

    all_spans = [span for cluster in clusters for span in cluster]
    #initialize new text being equal to old text
    new_text = doc.text
    start_positions = []
    end_positions = []
    all_replacements = []
    for cluster in clusters:
        noun_indices = get_span_noun_indices(doc, cluster)
        if len(noun_indices) > 0:
            mention_span, mention = get_cluster_head(doc, cluster, noun_indices)
            for coref in cluster:
                if coref != mention and not is_containing_other_spans(coref, all_spans):
                    # Execute the replacement
                    start_pos, end_pos = coref
                    # Replace the coref
                    start_positions.append(coref[0])
                    end_positions.append(coref[1])
                    # Store the replacement in a way that we can do it later
                    all_replacements.append((coref, mention_span))

    # Now do the replacements, take into account that the positions of the replacements change
    for idx, replacement in enumerate(all_replacements):
        start_pos, end_pos = start_positions[idx], end_positions[idx]
        _, mention_span = replacement

        begin_found, end_found = False, False
        tracker_begin, tracker_end = start_pos, end_pos
        while not (begin_found and end_found):
            if not begin_found:
                if new_text[tracker_begin] == '\n':
                    begin_found = True
                else:
                    if tracker_begin == 0:
                        begin_found = True
                    else:
                        tracker_begin -= 1
            if not end_found:
                if new_text[tracker_end] == '\n':
                    end_found = True
                else:
                    if tracker_end == len(new_text)-1:
                        end_found = True
                    else:
                        tracker_end += 1
        sentence = new_text[tracker_begin+1:tracker_end]
        start_pos_in_sentence = start_pos - tracker_begin - 1
        end_pos_in_sentence = end_pos - tracker_begin - 1
        mention_span = mention_span.lower()
        mention_span = mention_span.replace('.', '')
        mention_span = mention_span.replace(',', '')
        if new_text[start_positions[idx]-1] != ' ':
            if mention_span[0] != ' ':
                mention_span = ' ' + mention_span
        # if there is no space after, we add one. Be sure that we are not adding a space at the end of the text, this would cause an error as we would be out of range
        try:
            if end_positions[idx] < len(new_text)-1 and new_text[end_positions[idx]+1] != ' ':
                if mention_span[-1] != ' ':
                    mention_span = mention_span + ' '
        except:
            pass

        new_text = new_text[:start_positions[idx]] + mention_span + new_text[end_positions[idx]+1:]
        new_sentence = sentence[:start_pos_in_sentence] + mention_span + sentence[end_pos_in_sentence+1:]

        # write log
        write_log(sentence, new_sentence, 'Unknown', [start_pos_in_sentence], [end_pos_in_sentence], 'Coreference resolution', logger_coref)
        # Adapt the positions of the corefs, go over range idx until end
        for i in range(idx, len(all_replacements)):
            if start_positions[i] > start_positions[idx] and end_positions[i] > end_positions[idx]:
                # adapt start_position and end_position
                start_positions[i] = start_positions[i] - (end_positions[idx] - start_positions[idx] + 1) + len(mention_span)
                end_positions[i] = end_positions[i] - (end_positions[idx] - start_positions[idx] + 1) + len(mention_span)
                
    return new_text

def coreference_resolution(texts, logger_coref, batch_size=50000):
    """Resolve the coreferences in the texts using the fastcoref library

    Parameters:
        texts (list[str]): list of texts
        PATH_LOG (str): path to the log file
        batch_size (int): size of the batches to process the texts

    Return:
        new_texts (list[str]): list of texts with the coreferences resolved
    """

    new_texts = []
    # use tqdm to show progress bar
    for text in tqdm.tqdm(texts):
        # split up the text in batches of 200000 characters, split by \n
        if len(text) > batch_size:
            new_text = ''
            # we want to do coreference resolution on the text in batches of around 200000 characters, where we split by \n
            # we want to make sure that we do not split a sentence in half
            while len(text) > 50000:
                # find the position of the last \n before 200000 characters
                split_pos = text[:50000].rfind('\n')
                doc = nlp(text[:split_pos])
                clusters = doc._.coref_clusters
                new_text += replace_corefs(nlp(text[:split_pos]), logger_coref, clusters)
                text = text[split_pos:]
            doc = nlp(text)
            clusters = doc._.coref_clusters
            new_text += replace_corefs(doc, logger_coref, clusters)
            new_sentences = new_text
        else:
            doc = nlp(text)
            clusters = doc._.coref_clusters
            new_sentences = replace_corefs(doc, logger_coref, clusters)
    new_texts.append(new_sentences)

    return new_texts

def fix_line_breaks(texts, logger):
    """Fix line breaks in the texts

    Parameters:
        texts (list[str]): list of texts
        PATH_LOG (str): path to the log file

    Return:
        new_texts (list[str]): list of texts with the line breaks fixed
    """
    new_texts = []
    for idx, text in enumerate(texts):
        new_text = ''
        for idx, line in enumerate(text.split('\n')):
            # We start by fixing structures such as "beau- tiful" and "beau- tifully" to "beautiful" and "beautifully"
            # We also want to fix structures such as "beau-  tiful" to "beautiful", or "beau-   tiful" to "beautiful". 
            start_positions = [m.start() for m in re.finditer(r'(\w)-\s+(\w)', line)]
            end_positions = [m.end() for m in re.finditer(r'(\w)-\s+(\w)', line)]
            new_line = re.sub(r'(\w)-\s+(\w)', r'\1\2', line)
            # write log
            if len(start_positions) > 0:
                write_log(line, new_line, idx, start_positions, end_positions, 'Fixing line breaks', logger)
            new_text += new_line + '\n'
        new_texts.append(new_text)
    return new_texts

def main():
    """Convert pdfs to text and process the text"""
    ###################################   SETTINGS  ###################################################
    use_coref_resolution = False # whether to use coreference resolution

    # please run from the root directory of the project
    PATH_ROOT = Path.cwd()
    ####################################### FILL IN THE PATHS ########################################
    PATH_LOAD_TEXT = PATH_ROOT.joinpath('')
    PATH_SAVE_TEXT = PATH_ROOT.joinpath('') # path to the directory to save the processed text
    PATH_LOG = PATH_ROOT.joinpath('') # path to the log directory
    ##################################################################################################
    

    logger_general_output = get_logger('preprocessing_output', PATH_LOG + 'general_output_preprocessing.txt')
    logger_general_output.info('Current working directory: ' + PATH_ROOT)

    # check if folders exist, otherwise create them
    if not PATH_LOG.exists():
        logger_general_output.info('Creating log folder...')
        Path.mkdir(PATH_LOG)

    if not PATH_SAVE_TEXT.exists():
        logger_general_output.info('Creating folder to save texts... ')
        Path.mkdir(PATH_SAVE_TEXT)

    PATH_LOG_CITATIONS = PATH_LOG.joinpath('log_citations.txt')
    PATH_LOG_ABBREVIATIONS = PATH_LOG.joinpath('log_abbreviations.txt')
    PATH_LOG_COREFERENCE = PATH_LOG.joinpath('log_coreference.txt')
    PATH_LOG_LINE_BREAKS = PATH_LOG.joinpath('log_line_breaks.txt')
    # create the loggers using the logging module
    logger_citations = get_logger('citations', PATH_LOG_CITATIONS)
    logger_abbreviations = get_logger('abbreviations', PATH_LOG_ABBREVIATIONS)
    logger_coreference = get_logger('coreference', PATH_LOG_COREFERENCE)
    logger_line_breaks = get_logger('line_breaks', PATH_LOG_LINE_BREAKS)


    texts, text_file_names = get_text_files_in_dir(PATH_LOAD_TEXT)

    # ------------------- FIX LINE BREAKS -------------------

    logger_general_output.info('Fixing line breaks...')
    texts = fix_line_breaks(texts, logger_line_breaks)

    # ------------------- REMOVE CITATIONS -------------------
    logger_general_output.info('Removing citations...')
    texts = remove_citations(texts, logger_citations)

    # ------------------- EXPAND ABBREVIATIONS -------------------
    logger_general_output.info('Expanding abbreviations...')
    texts, errors_with_abbreviations = expand_abbreviations(texts, logger_abbreviations)
    if len(errors_with_abbreviations) > 0:
        logger_general_output.info('Errors with the following abbreviations: ' + str(errors_with_abbreviations))

    # ------------------- COREFERENCE RESOLUTION -------------------
 
    if use_coref_resolution:
        # record start time
        start_time = time.time()
        logger_general_output.info('Resolving coreferences...')
        texts = coreference_resolution(texts, logger_coreference)
        # record end time
        end_time = time.time()
        logger_general_output.info('Time for coreference resolution: ' + str(end_time - start_time) + ' seconds')
    # ------------------- SAVE TEXT -------------------
    logger_general_output.info('Saving text...')
    for i, text in enumerate(texts):
        new_path = Path.joinpath(PATH_SAVE_TEXT, text_file_names[i])
        with open(new_path, 'w', encoding='utf-8') as f:
            f.write(text)
    logger_general_output.info('Done!')


if __name__ == "__main__":
    main()