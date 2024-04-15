from helpers import get_logger
import pandas as pd
import numpy as np
import spacy

import pickle
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster

import time
from transformers import AutoModel
from transformers import AutoTokenizer

import tqdm
from collections import Counter
import itertools
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
nlp = spacy.load("en_core_web_lg")

def add_paper_id(df):
    """ Add the paper id to the triplets
    
        Parameters: 
            df (pd.DataFrame): the dataframe with the triplets

        Returns:
            df (pd.DataFrame): the dataframe with the paper id added to the triplets
    """
    for idx, row in df.iterrows():
        triplets = row['triplets']
        # remove duplicates
        triplets = list(set(triplets))
        new_triplets = []
        for triplet in triplets:
            new_triplet = (triplet[0], triplet[1], triplet[2], row['paper'])
            new_triplets.append(new_triplet)
        df.at[idx, 'triplets'] = new_triplets
    return df

def load_triplets(path):
    """ Load the triplets from a csv file

        Parameters:
            path (str): the path to the csv file

        Returns:
            df (pd.DataFrame): the dataframe with the triplets
    """
    df = pd.read_csv(path)
    df['triplets'] = df['triplets'].apply(eval)
    df = add_paper_id(df)
    return df

def filter_generic_triplets(df, path_word_counts, threshold):
    """ Filter out the generic triplets from the dataframe

        Parameters:
            df (pd.DataFrame): the dataframe with the triplets
            path_word_counts (str): the path to the word counts
            threshold (int): the threshold for the word counts

        Returns:
            df (pd.DataFrame): the dataframe with the generic triplets removed
    """
    # Load word counts
    with open(path_word_counts, 'rb') as f:
        word_counts = pickle.load(f)

    for idx, row in df.iterrows():
        triplets = row['triplets']
        # keep the triplets where word_counts[subject] < threshold or where subject is not in word_counts
        triplets = [triplet for triplet in triplets if word_counts[triplet[0]] < threshold or triplet[0] not in word_counts]
        df.at[idx, 'triplets'] = triplets
    # Remove rows with empty triplets
    df = df[df['triplets'].map(len) > 0]
    return df

def get_subjverbobj(df):
    """ Get the subjects, verbs and objects from the dataframe

        Parameters:
            df (pd.DataFrame): the dataframe with the triplets

        Returns:
            subjects (list): the list of subjects
            verbs (list): the list of verbs
            objects (list): the list of objects
    """
    subjects = set()
    objects = set()
    verbs = set()
    for _, row in df.iterrows():
        triplets_col = row['triplets']
        for triplet in triplets_col:
            subject, verb, object, _ = triplet
            subjects.add(subject)
            objects.add(object)
            verbs.add(verb)
    return list(subjects),  list(verbs), list(objects)

def get_triplets_list(df):
    """ Get the triplets from the dataframe

        Parameters:
            df (pd.DataFrame): the dataframe with the triplets

        Returns:
            triplets (list): the list of triplets
    """
    triplets = set()
    for _, row in df.iterrows():
        triplets_col = row['triplets']
        for triplet in triplets_col:
            triplets.add(triplet)
    return list(triplets)

def get_subjobjverb_embeds(subjects_all, objects_all, verbs_all, logger):
    """ Get the embeddings for the subjects, objects and verbs

        Parameters:
            subjects_all (list): the list of subjects
            objects_all (list): the list of objects 
            verbs_all (list): the list of verbs
            logger (logging.Logger): the logger

        Returns:
            subject_embeds (dict): the dictionary with the subjects and their embeddings
            verb_embeds (dict): the dictionary with the verbs and their embeddings
            object_embeds (dict): the dictionary with the objects and their embeddings
    """
    object_embeds = {}
    subject_embeds = {}
    verb_embeds = {}
    for i, obj in enumerate(objects_all):
        if i % 100 == 0:
            logger.info(i, ' out of ', len(objects_all))
        input_ids = tokenizer.encode(obj, return_tensors='pt')
        outputs = model(input_ids)
        object_embeds[obj] = outputs.pooler_output.detach().numpy()
    for i, sub in enumerate(subjects_all):
        if i % 100 == 0:
            logger.info(i, ' out of ', len(subjects_all))
        input_ids = tokenizer.encode(sub, return_tensors='pt')
        outputs = model(input_ids)
        subject_embeds[sub] = outputs.pooler_output.detach().numpy()
    for i, verb in enumerate(verbs_all):
        if i % 100 == 0:
            logger.info(i, ' out of ', len(verbs_all))
        input_ids = tokenizer.encode(verb, return_tensors='pt')
        outputs = model(input_ids)
        verb_embeds[verb] = outputs.pooler_output.detach().numpy()
    return subject_embeds,  verb_embeds, object_embeds

def load_embeds(path_subject_embed, path_object_embed, path_verb_embed, subjects_all, objects_all, verbs_all, logger):
    """ Load the embeddings for the subjects, objects and verbs
    
        Parameters:
            path_subject_embed (str): the path to the subject embeddings
            path_object_embed (str): the path to the object embeddings
            path_verb_embed (str): the path to the verb embeddings
            subjects_all (list): the list of subjects
            objects_all (list): the list of objects
            verbs_all (list): the list of verbs
            logger (logging.Logger): the logger

        Returns:    
            subject_embeds (dict): the dictionary with the subjects and their embeddings
            verb_embeds (dict): the dictionary with the verbs and their embeddings
            object_embeds (dict): the dictionary with the objects and their embeddings
    """
    if not path_subject_embed.exists() or not path_object_embed.exists() or not path_verb_embed.exists():
        logger.info('Embeddings not found, computing them')
        subject_embeds, verb_embeds, object_embeds  = get_subjobjverb_embeds(subjects_all, objects_all, verbs_all, logger)
        with open(path_subject_embed, 'wb') as f:
            pickle.dump(subject_embeds, f)
        with open(path_object_embed, 'wb') as f:
            pickle.dump(object_embeds, f)
        with open(path_verb_embed, 'wb') as f:
            pickle.dump(verb_embeds, f)

    else:
        logger.info('Embeddings found, loading them')
        with open(path_subject_embed, 'rb') as f:
            subject_embeds = pickle.load(f)
        with open(path_object_embed, 'rb') as f:
            object_embeds = pickle.load(f)
        with open(path_verb_embed, 'rb') as f:
            verb_embeds = pickle.load(f)

    return subject_embeds, object_embeds, verb_embeds

def get_cluster_sub_dict(clusters_sub, subjects):
    """ Get the dictionary with the subjects and their clusters

        Parameters:
            clusters_sub (np.array): the clusters of the subjects
            subjects (list): the list of subjects

        Returns:
            clusters_sub_dict (dict): the dictionary with the subjects and their clusters
    """
    clusters_sub_dict = {}
    for i, cluster in enumerate(clusters_sub):
        subject = subjects[i]
        clusters_sub_dict[subject] = cluster
    return clusters_sub_dict

def get_cluster_obj_dict(clusters_obj, objects):
    """ Get the dictionary with the objects and their clusters

        Parameters:
            clusters_obj (np.array): the clusters of the objects
            objects (list): the list of objects

        Returns:
            clusters_obj_dict (dict): the dictionary with the objects and their clusters
    """
    clusters_obj_dict = {}
    for i, cluster in enumerate(clusters_obj):
        obj = objects[i]
        clusters_obj_dict[obj] = cluster
    return clusters_obj_dict


def cluster_based_on_subj(triplets, clusters_sub, logger):
    """ Cluster the triplets based on the subjects

        Parameters:
            triplets (list): the list of triplets
            clusters_sub (dict): the dictionary with the subjects and their clusters
            logger (logging.Logger): the logger

        Returns:
            clustered_triplets (dict): the dictionary with the clusters and the triplets
    """
    clustered_triplets = {}
    for i, triplet in enumerate(triplets):
        subj, verb, obj, paper = triplet
        try:
            subj_cluster = clusters_sub[subj]
        except:
            logger.info('Doesnt work for', subj)
            continue
        subj_cluster = clusters_sub[subj]
        if subj_cluster not in clustered_triplets:
            clustered_triplets[subj_cluster] = []
        clustered_triplets[subj_cluster].append(triplet)
    return clustered_triplets

def cluster_based_on_obj(triplets, clusters_obj, logger):
    """ Cluster the triplets based on the objects

        Parameters:
            triplets (list): the list of triplets
            clusters_obj (dict): the dictionary with the objects and their clusters
            logger (logging.Logger): the logger

        Returns:
            clustered_triplets (dict): the dictionary with the clusters and the triplets
    """
    clustered_triplets = {}
    for i, triplet in enumerate(triplets):
        subj, verb, obj, paper = triplet
        try:
            obj_cluster = clusters_obj[obj]
        except:
            logger.info('Doesnt work for', obj)
            continue
        if obj_cluster not in clustered_triplets:
            clustered_triplets[obj_cluster] = []
        clustered_triplets[obj_cluster].append(triplet)
    return clustered_triplets




def make_clusters_scibert(subjects_emb, objects_emb, subset=1):
    """ Make the clusters of the subjects and objects based on the embeddings

        Parameters:
            subjects_emb (dict): the dictionary with the subjects and their embeddings
            objects_emb (dict): the dictionary with the objects and their embeddings
            subset (float): the fraction of the embeddings to use for clustering

        Returns:
            Z_sub (np.array): the linkage matrix for the subjects
            Z_obj (np.array): the linkage matrix for the objects
            subjects_subset (np.array): the subset of subjects
            objects_subset (np.array): the subset of objects
    """

    embeddings_sub = np.array(list(subjects_emb.values()))
    embeddings_obj = np.array(list(objects_emb.values()))
    subjects_ordered = np.array(list(subjects_emb.keys()))
    objects_ordered = np.array(list(objects_emb.keys()))

    # take random subset
    random_indices_sub = np.random.choice(len(embeddings_sub), int(len(embeddings_sub) * subset), replace=False)
    random_indices_obj = np.random.choice(len(embeddings_obj), int(len(embeddings_obj) * subset), replace=False)
    embeddings_sub = embeddings_sub[random_indices_sub]
    embeddings_obj = embeddings_obj[random_indices_obj]
    subjects_subset = subjects_ordered[random_indices_sub]
    objects_subset = objects_ordered[random_indices_obj]

    # Now the shape is (n, 1, 768), we need to reshape it to (n, 768)
    embeddings_sub = np.array([embedding[0] for embedding in embeddings_sub])
    embeddings_obj = np.array([embedding[0] for embedding in embeddings_obj])
    
    # make the clusters
    Z_sub = linkage(embeddings_sub, 'average', metric='cosine')
    Z_obj = linkage(embeddings_obj, 'average', metric='cosine')
    return Z_sub, Z_obj, subjects_subset, objects_subset


def get_clustered_triplets(triplets, clusters_sub, clusters_obj, subjects, objects, logger):
    """ Get the triplets together that have subjects of the same cluster and objects of the same cluster

        Parameters:
            triplets (list): the list of triplets
            clusters_sub (np.array): the clusters of the subjects
            clusters_obj (np.array): the clusters of the objects
            subjects (list): the list of subjects
            objects (list): the list of objects
            logger (logging.Logger): the logger

        Returns:
            triplets_clustered (dict): the dictionary with the clusters and the triplets
    """
    # get the triplets together that have subjects of the same cluster and objects of the same cluster
    triplets_clustered = {}
    for i, triplet in enumerate(triplets):
        subject, verb, obj, paper = triplet
        try:
            index_sub = np.where(subjects == subject)[0][0]
            index_obj = np.where(objects == obj)[0][0]
        except:
            logger.info('Error with triplet: ', triplet)
            continue
        cluster_sub = clusters_sub[index_sub]
        cluster_obj = clusters_obj[index_obj]
        if (cluster_sub, cluster_obj) not in triplets_clustered:
            triplets_clustered[(cluster_sub, cluster_obj)] = []
        triplets_clustered[(cluster_sub, cluster_obj)].append(triplet)
    return triplets_clustered

def get_fair_groups(triplets_clustered, cat_dict, one_triplet_per_paper=True):
    """ Get the fair groups of triplets

        Parameters:
            triplets_clustered (dict): the dictionary with the clusters and the triplets
            cat_dict (dict): the dictionary with the papers and their categories
            one_triplet_per_paper (bool): whether to have maximum one triplet per paper
            balanced_mixed_groups (bool): whether to have balanced mixed groups

        Returns:
            groups_cs (dict): the dictionary with the number of triplets and the groups of triplets that are cs
            groups_quant (dict): the dictionary with the number of triplets and the groups of triplets that are quantum
            groups_csquant (dict): the dictionary with the number of triplets and the groups of triplets that are mixed
    """
    groups_cs = {}
    groups_quant = {}
    groups_csquant = {}

    if one_triplet_per_paper:
        groups_one_triplet_per_paper = []
        for _, triplets in tqdm.tqdm(triplets_clustered.items()):
            # there may be multiple triplets from the same paper, we want maximum one per paper. Get all sets of triplets that have maximum one triplet per paper
            triplets_per_paper = {}
            for triplet in triplets:
                paper = triplet[3]
                if paper in triplets_per_paper:
                    triplets_per_paper[paper].append(triplet)
                else:
                    triplets_per_paper[paper] = [triplet]
            # Now get all possible combinations of triplets that have maximum one triplet per paper
            all_keys = list(triplets_per_paper.keys())
            value_combinations = [v for v in itertools.product(*[triplets_per_paper[k] for k in all_keys])]
            groups_one_triplet_per_paper.extend(value_combinations)

    else:
        groups_one_triplet_per_paper = triplets_clustered.values()

    for group in tqdm.tqdm(groups_one_triplet_per_paper):
        quantum_trips = []
        cs_trips = []
        for triplet in group:
            paper = triplet[3]
            if cat_dict[paper] == 'cs':
                cs_trips.append(triplet)
            else:
                quantum_trips.append(triplet)
        if len(cs_trips) > 1:
            if len(cs_trips) in groups_cs:
                groups_cs[len(cs_trips)].append(cs_trips)
            else:
                groups_cs[len(cs_trips)] = [cs_trips]
        if len(quantum_trips) > 1:
            if len(quantum_trips) in groups_quant:
                groups_quant[len(quantum_trips)].append(quantum_trips)
            else:
                groups_quant[len(quantum_trips)] = [quantum_trips]
        if len(cs_trips) > 0 and len(quantum_trips) > 0:
            # make sure that at least 25 percent of the triplets are from the minority group
            total_triplets = len(cs_trips) + len(quantum_trips)
            if len(cs_trips) / total_triplets > 0.25 and len(quantum_trips) / total_triplets > 0.25:
                if len(group) in groups_csquant:
                    groups_csquant[len(group)].append(group)
                else:
                    groups_csquant[len(group)] = [group]
    return groups_cs, groups_quant, groups_csquant

def get_category_dictionary(path_cs, path_quant):
    """ Get the dictionary with the papers and their categories

        Parameters:
            path_cs (str): the path to the cs papers
            path_quant (str): the path to the quantum papers

        Returns:
            cat_dict (dict): the dictionary with the papers and their categories
    """

    cat_dict = {}
    # the titles in path_cs get value 'cs'
    for file in path_cs.rglob('*.pdf'):
        #change the .pdf to .txt
        title = file[:-4] + '.txt'
        cat_dict[title] = 'cs'
    # the titles in path_quant get value 'quant'
    for file in path_quant.rglob('*.pdf'):
        #change the .pdf to .txt
        title = file[:-4] + '.txt'
        cat_dict[title] = 'quant'
    return cat_dict

def compute_variances(fair_group, embeds, cat_dict, clustered_by='subj_obj', method='mean', mixed_groups=False):
    """ Compute the variances for the fair groups

        Parameters:
            fair_group (dict): the dictionary with the fair groups
            embeds (dict): the dictionary with the embeddings
            cat_dict (dict): the dictionary with the papers and their categories
            clustered_by (str): whether to cluster by subject, object or both
            method (str): the method to use for computing the variance
            mixed_groups (bool): whether to have mixed groups

        Returns:
            distances (dict): the dictionary with the variances
    """
    distances = {}
    for size, groups in tqdm.tqdm(fair_group.items()):
        for group in groups:
            if mixed_groups:
                cs_embeds = []
                quant_embeds = []
            else:
                all_embeds = []
            for triplet in group:
                embed_of = 0 if clustered_by == 'obj' else 2 if clustered_by == 'subj' else 1
                word_to_embed = triplet[embed_of]
                category = cat_dict[triplet[3]]
                if word_to_embed in embeds:
                    if mixed_groups:
                        if category == 'cs':
                            cs_embeds.append(embeds[word_to_embed])
                        else:
                            quant_embeds.append(embeds[word_to_embed])
                    else:
                        all_embeds.append(embeds[word_to_embed])


            if mixed_groups:
                cs_embeds = np.array(cs_embeds)
                quant_embeds = np.array(quant_embeds)
                cs_embeds = np.squeeze(cs_embeds, axis=1)
                quant_embeds = np.squeeze(quant_embeds, axis=1)
            else:
                all_embeds = np.array(all_embeds)
                all_embeds = np.squeeze(all_embeds, axis=1)
            
            if mixed_groups:
                # We want to compute the mean distance between the cs and quantum vectors
                # iterate over rows of cs_embeds and compute the distance to each row of quant_embeds
                distances_csquant = []
                for i in range(cs_embeds.shape[0]):
                    cs_vec = cs_embeds[i, :]
                    for j in range(quant_embeds.shape[0]):
                        quant_vec = quant_embeds[j, :]
                        distances_csquant.append(np.linalg.norm(cs_vec - quant_vec))
                if method == 'mean':
                    agg_distance = np.mean(distances_csquant)
                elif method == 'third_quartile':
                    agg_distance = np.percentile(distances_csquant, 75)
                    
                if size in distances:
                    distances[size].append(agg_distance)
                else:
                    distances[size] = [agg_distance]
            else:
                distances_temp = []
                for i in range(all_embeds.shape[0]):
                    vec = all_embeds[i, :]
                    for j in range(all_embeds.shape[0]):
                        if i != j:
                            vec2 = all_embeds[j, :]
                            distances_temp.append(np.linalg.norm(vec - vec2))
                if method == 'mean':
                    agg_distance = np.mean(distances_temp)
                elif method == 'third_quartile':
                    agg_distance = np.percentile(distances_temp, 75)
                if size in distances:
                    distances[size].append(agg_distance)
                else:
                    distances[size] = [agg_distance]

                    
            
    for size, distance in distances.items():
        distances[size] = np.mean(distance)
    return distances

def compute_cluster_diameter(fair_groups_cs, fair_groups_quant, fair_groups_csquant, cat_dict, verb_embeds, method='mean', clustered_by='subj_obj'):
    """ Compute the cluster diameter for the fair groups

        Parameters:
            fair_groups_cs (dict): the dictionary with the cs fair groups
            fair_groups_quant (dict): the dictionary with the quantum fair groups
            fair_groups_csquant (dict): the dictionary with the mixed fair groups
            cat_dict (dict): the dictionary with the papers and their categories
            verb_embeds (dict): the dictionary with the verb embeddings
            method (str): the method to use for computing the variance
            clustered_by (str): whether to cluster by subject, object or both

        Returns:
            cs_diameter (dict): the dictionary with the cs cluster diameters
            quant_diameter (dict): the dictionary with the quantum cluster diameters
            csquant_diameter (dict): the dictionary with the mixed cluster diameters
    """
    cs_variances = compute_variances(fair_groups_cs, verb_embeds, cat_dict=cat_dict, method=method, clustered_by=clustered_by)
    quant_variances = compute_variances(fair_groups_quant, verb_embeds, cat_dict=cat_dict, method=method, clustered_by=clustered_by)
    csquant_variances = compute_variances(fair_groups_csquant, verb_embeds, cat_dict=cat_dict, method=method, clustered_by=clustered_by, mixed_groups=True)
    return cs_variances, quant_variances, csquant_variances

def get_counts(path):
    """ Get the counts of the words in the files in the folder.

        Parameters
            path (str): the path to the folder with the files

        Returns
            counter (Counter): a Counter object with the counts of the words
    """
    counter = Counter()
    #iterate over the files in the folder using tqdm
    for file in path.rglob('*.txt'):
        #read the file
        with open(path.joinpath(file), 'r', encoding='utf-8') as f:
            text = f.read()
            #split the text into words
            words = text.split()
            # for every word that appears over 5 times, increment the count
            unique_words = set(words)
            for word in unique_words:
                if words.count(word) > 5:
                    if word in counter:
                        counter[word] += 1
                    else:
                        counter[word] = 1
    return counter


def main():
    cwd = Path.cwd()

    ############################ SETTINGS ############################
    THRESHOLD_COUNTS = 5
    CLUSTER_THRESHOLD_SUB = 0.05
    CLUSTER_THRESHOLD_OBJ = 0.1
    CLUSTER_BY = 'subj_obj'

    PATH_TO_WORKING_FOLDER = cwd.joinpath('') # Put here the folder where the triplets are stored, the embeddings and clusters will be stored here as well
    PATH_TO_FILES_FOR_COUNTS = cwd.joinpath('')  # Put here the folder with the files for the word counts, these files should be from different categories than the target categories
    PATH_LOG = cwd.joinpath('') # Put here the path to the log folder
    #################################################################

    PATH_TRIPLETS = PATH_TO_WORKING_FOLDER.join('processed_triplets.csv')  # place where the triplets are stored
    PATH_ARXIV_COUNTS = PATH_TO_WORKING_FOLDER.join('word_counts.pkl') # place where the word counts are stored
    PATH_SUBJ_EMB = PATH_TO_WORKING_FOLDER.join('subjects_embeds.pkl') # place where the object embeddings are stored
    PATH_OBJ_EMB = PATH_TO_WORKING_FOLDER.join('objects_embeds.pkl') # place where the subject embeddings are stored
    PATH_VERB_EMBEDS = PATH_TO_WORKING_FOLDER.join('verbs_embeds.pkl') # place where the verb embeddings are stored
    PATH_SAVE_CLUSTERS = PATH_TO_WORKING_FOLDER.join('clusters.pkl') # place where the clusters are stored
    
    general_logger = get_logger(PATH_LOG.joinpath('general_logger_clustering.txt'))

    # check if the counts are already saved
    if not PATH_ARXIV_COUNTS.exists():
        # get the counts
        counts = get_counts(PATH_TO_FILES_FOR_COUNTS)
        # save the counts
        with open(PATH_ARXIV_COUNTS, 'wb') as f:
            pickle.dump(counts, f)

    # Load the triplets
    df = load_triplets(PATH_TRIPLETS)

    # Filter out the generic triplets by using word counts
    df = filter_generic_triplets(df, PATH_ARXIV_COUNTS, THRESHOLD_COUNTS)

    # Get the subjects and objects, and get all the triplets
    subjects,  verbs, objects = get_subjverbobj(df)
    triplets = get_triplets_list(df)
    
    # Get the embeddings, then filter to only include the ones in the triplets
    subjects_embeds, objects_embeds, verb_embeds = load_embeds(PATH_SUBJ_EMB, PATH_OBJ_EMB, PATH_VERB_EMBEDS, subjects, objects, verbs, general_logger)

    # Cluster the embeddings
    Z_sub, Z_obj, subjects, objects = make_clusters_scibert(subjects_embeds, objects_embeds, subset=1)

    # Cut the clusters according to the threshold
    clusters_sub = fcluster(Z_sub, CLUSTER_THRESHOLD_SUB, criterion='distance')
    clusters_obj = fcluster(Z_obj, CLUSTER_THRESHOLD_OBJ, criterion='distance')
    
    # Convert the clusters to dictionaries
    clusters_sub_dict = get_cluster_sub_dict(clusters_sub, subjects)
    clusters_obj_dict = get_cluster_obj_dict(clusters_obj, objects)

    # Cluster the triplets based on the setting
    if CLUSTER_BY == 'subj_obj':
        clustered_triplets = get_clustered_triplets(triplets, clusters_sub, clusters_obj, subjects, objects, general_logger)
    if CLUSTER_BY == 'subj':
        clustered_triplets= cluster_based_on_subj(triplets, clusters_sub_dict, general_logger)
    if CLUSTER_BY == 'obj':
        clustered_triplets = cluster_based_on_obj(triplets, clusters_obj_dict, general_logger)

    # Save the clusters
    with open(PATH_SAVE_CLUSTERS, 'wb') as f:
        pickle.dump(clustered_triplets, f)

if __name__ == "__main__":
    main()
