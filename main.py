### Driver code for distributional lesk with removing stopwords from context and gloss

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from gensim.models import KeyedVectors

from loader import *
from dict_utilities import *
from helper import *

import numpy as np
from numpy.linalg import norm
import json

import gensim.downloader as api
from tqdm import tqdm

SEMCOR_DATA_FILE = './SemCor/semcor.data.xml'
SEMCOR_LABELLED = './SemCor/semcor.gold.key.txt'
SENSEVAL_2_DATA_FILE = './senseval2/senseval2.data.xml'
SENSEVAL_2_LABELLED = './senseval2/senseval2.gold.key.txt'
SENSEVAL_3_DATA_FILE = './senseval3/senseval3.data.xml'
SENSEVAL_3_LABELLED = './senseval3/senseval3.gold.key.txt'

word_embeddings = api.load('word2vec-google-news-300')
nltk.download('stopwords')
StopWords = stopwords.words("english")

def get_embed(lst_strings):
  """
  Returns the embedding for the given list of strings --
  average of the word embeddings of each word in the list
  """
  #lst_strings = remove_stopwords(lst_strings) ## remove stopwords from the list
  vectors_strings = []
  for word in lst_strings:
    try:
        vectors_strings.append(word_embeddings[word])
    except:
      pass
  embed = np.mean(vectors_strings, axis=0)

  return embed

def remove_stopwords(lst_strings):
    """
    Remove the stopwords from the list of strings
    :param lst_strings:
    :return:
    """
    lst_strings = [word for word in lst_strings if word not in StopWords]
    return lst_strings

def dist_lesk(lemmas, labels, mapping_dict, lexeme_embeds, synset_embeds):
    """
    Trains the distributional lesk algorithm with semcor dataset
    The crucial part of the algorithm is to replace word_embeds with synset_embeds for the words
    which have been disambiguated.
    The function also find the accuracy of distributional lesk
    """
    correct_count = 0
    total = len(lemmas)

    for lemma_id, lemma_label in tqdm(labels.items()):
        ## Filter the lemmas which don't have more than one synset in their definitions
        this_wsd_inst = lemmas[lemma_id]
        if len(wn.synsets(this_wsd_inst.lemma)) <= 1:
            continue

        ## Get the context embeds
        context_embed = get_embed(this_wsd_inst.context)
        final_score = 0
        final_synset = wn.synsets(this_wsd_inst.lemma)[0] ## Just initialize it with the first synset
        final_wn_synset_id = ''
        for this_synset in wn.synsets(this_wsd_inst.lemma):
            print(this_synset)
            ## Computations for this synset-lemma pair
            ## Get the gloss embeds
            gloss_embed = get_embed(this_synset.definition().split(" "))
            ## This key is formed to get the lexeme embedding
            wn_synset_id = ''
            for synset_lemma in this_synset.lemmas():
                this_synset_key = synset_lemma.key()
                ## Get the wn-id from mapping.txt and using above dictionary
                if this_synset_key in mapping_dict.keys():
                    wn_synset_id = mapping_dict[this_synset_key]
                    break

            if wn_synset_id != '':
                ## Get the lexeme embeds using wn_id
                if wn_synset_id in lexeme_embeds.keys():
                    lexeme_embed = lexeme_embeds[wn_synset_id]
                    ## Find similarity score for each word, sense pair
                    score = np.dot(lexeme_embed, context_embed) / (norm(lexeme_embed) * norm(context_embed)) + \
                            np.dot(gloss_embed, context_embed) / (norm(gloss_embed) * norm(context_embed))
                else:
                    ## If no lexeme available then simply take the similarity of gloss and context
                    score = np.dot(gloss_embed, context_embed) / (norm(gloss_embed) * norm(context_embed))

                ## Write the code for finding the maximum score
                try:
                    if score > final_score:
                        final_score = score
                        final_wn_synset_id = wn_synset_id
                        final_synset = this_synset
                except:
                    print("THE SCORE IS NAN  **********************************")

        print("The maximum score is:", final_score)
        print("THE FINAL PREDICTED SYNSET IS: ", final_synset)
        ## The disambiguated lemma is updated with the synset embedding
        try:
            word_embeddings[this_wsd_inst.lemma] = synset_embeds[final_wn_synset_id]
            print(" WORD EMBEDDINGS SUCCCCESSSSSSFULLY UPDATED WITH SYSNETS EMBEDDING")
        except:
            pass

        ## Call the eval function for finding accuracy
        correct_label = labels[lemma_id][0]
        if eval_acc(final_synset, correct_label):
            correct_count += 1

    return (correct_count/total)*100

if __name__ == '__main__':
    ## Lemmas are returned in key, value where key is lemma_id and value is WSDInstance (lemma_id, sentence_id, lemma, context, index, no_synsets)
    semcor_lemmas = sort_lemmas(load_instances(SEMCOR_DATA_FILE))
    semcor_labels = get_labels(SEMCOR_LABELLED)

    senseval_2_lemmas = sort_lemmas(load_instances(SENSEVAL_2_DATA_FILE))
    senseval_2_labels = get_labels(SENSEVAL_2_LABELLED)

    senseval_3_lemmas = sort_lemmas(load_instances(SENSEVAL_3_DATA_FILE))
    senseval_3_labels = get_labels(SENSEVAL_3_LABELLED)

    ## Driver code for distributional lesk
    ## Select 5000 sentences
    #semcor_5000_lemmas = sort_lemmas(get_semcor_data(semcor_lemmas))

    mapping_dict = read_mapping()
    lexeme_embeds = read_lexeme_embeds()
    synset_embeds = read_synset_embeds()

    ## Driver code for evaluation of distributional lesk
    semcor_labels = dict(list(semcor_labels.items())[:2000])
    sem_accuracy = dist_lesk(semcor_lemmas, semcor_labels, mapping_dict, lexeme_embeds, synset_embeds)
    print("Dist_lesk accuracy on Semcor:", sem_accuracy)

    #senseval_2_accuracy = dist_lesk(senseval_2_lemmas, senseval_2_labels, mapping_dict, lexeme_embeds, synset_embeds)
    #print("Dist_lesk accuracy on Senseval_2:", senseval_2_accuracy)

    #senseval_3_accuracy = dist_lesk(senseval_3_lemmas, senseval_3_labels, mapping_dict, lexeme_embeds, synset_embeds)
    #print("Dist_lesk accuracy on Senseval_3:", senseval_3_accuracy)

    word_embeddings.save("word_embeddings.kv")
    word_embeddings = KeyedVectors.load("word_embeddings.kv")