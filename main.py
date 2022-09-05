### Driver code for distributional lesk

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

from loader import *
from dict_utilities import *
from helper import *

import numpy as np
from numpy.linalg import norm

import gensim.downloader as api
from tqdm import tqdm

SEMCOR_DATA_FILE = './semCor/semcor.data.xml'
SEMCOR_LABELLED = './semCor/semcor.gold.key.txt'
SENSEVAL_2_DATA_FILE = './senseval2/senseval2.data.xml'
SENSEVAL_2_LABELLED = './senseval2/senseval2.gold.key.txt'
SENSEVAL_3_DATA_FILE = './senseval3/senseval3.data.xml'
SENSEVAL_3_LABELLED = './senseval3/senseval3.gold.key.txt'

word_embeddings = api.load('word2vec-google-news-300')
#word_embeddings = np.load('embeddings.npy')
StopWords = stopwords.words("english")

def get_embed(lst_strings):
  """
  Returns the embedding for the given list of strings --
  average of the word embeddings of each word in the list
  """
  lst_strings = remove_stopwords(lst_strings)
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


def train_dist_lesk(lemmas, mapping_dict, lexeme_embeds, synset_embeds):
    """
    Trains the distributional lesk algorithm with semcor dataset
    The crucial part of the algorithm is to replace word_embeds with synset_embeds for the words
    which have been disambiguated.
    """
    for lemma_id, wsd_inst in tqdm(lemmas.items()):
        ## Filter the lemmas which don't have more than one synset in their definitions
        if len(wn.synsets(wsd_inst.lemma)) <= 1:
            continue
        ## Get the context embeds
        context_embed = get_embed(wsd_inst.context)
        final_score = 0
        final_synset_keys = ''
        final_wn_synset_id = ''
        for synset in wn.synsets(wsd_inst.lemma):
            ## Computations for this synset-lemma pair
            ## Get the gloss embeds
            gloss_embed = get_embed(synset.definition().split(" "))
            ## This key is formed to get the lexeme embedding
            this_synset_key = ''
            for synset_lemma in synset.lemmas():
                this_synset_key += synset_lemma.key()
                this_synset_key += ','
            #this_synset_key = this_synset_key[:-1]  # Remove the last comma

            ## Get the wn-id from mapping.txt and using above dictionary
            try:
                wn_synset_id = mapping_dict[this_synset_key]
                ## Get the lexeme embeds using wn_id
                lexeme_embed = lexeme_embeds[wn_synset_id]
                ## Find similarity score for each word, sense pair
                score = np.dot(lexeme_embed, context_embed) / (norm(lexeme_embed) * norm(context_embed)) + \
                        np.dot(gloss_embed, context_embed) / (norm(gloss_embed) * norm(context_embed))

                ## Write the code for finding the maximum score
                if score > final_score:
                    final_score = score
                    final_synset_keys = this_synset_key
                    final_wn_synset_id = wn_synset_id
            except:
                pass

        print("The maximum score is:", final_score)
        print("The predicted synset keys are:", final_synset_keys)
        ## The disambiguated lemma is updated with the synset embedding
        try:
            word_embeddings[wsd_inst.lemma] = synset_embeds[final_wn_synset_id]
            print("-----------------SUCCCCESSSSSS----------------------")
        except:
            pass

    return


def eval_dist_lesk(lemmas, labels, mapping_dict, lexeme_embeds):
    """
    Evaluation for distributional lesk
    """
    correct_count = 0
    total = len(labels)

    for lemma_id, lemma_label in tqdm(labels.items()):
        this_wsd_inst = lemmas[lemma_id]

        ## Get the context embeds
        context_embed = get_embed(this_wsd_inst.context)
        final_score = 0
        final_synset_keys = ''
        for synset in wn.synsets(this_wsd_inst.lemma):
            ## Computations for this synset, lemma pair
            ## Get the gloss embeds
            gloss_embed = get_embed(synset.definition().split(" "))
            exit()
            this_synset_key = ''
            for synset_lemma in synset.lemmas():
                this_synset_key += synset_lemma.key()
                this_synset_key += ','
            #this_synset_key = this_synset_key[:-1]  # Remove the last comma

            ## Get the wn-id from mapping.txt and using above dictionary
            try:
                wn_synset_id = mapping_dict[this_synset_key]
                ## Get the lexeme embeds using wn_id
                lexeme_embed = lexeme_embeds[wn_synset_id]
                ## Find similarity score for each word, sense pair
                score = np.dot(lexeme_embed, context_embed) / (norm(lexeme_embed) * norm(context_embed)) + \
                        np.dot(gloss_embed, context_embed) / (norm(gloss_embed) * norm(context_embed))

                if score > final_score:
                    final_score = score
                    final_synset_keys = this_synset_key
            except:
                pass
        ## Write the code for finding the maximum score
        print("The maximum score is:", final_score)
        print("The predicted synset keys are:", final_synset_keys)

        correct_label = labels[lemma_id][0]
        pred_label = final_synset_keys.split(',')

        for prediction in pred_label:
            if correct_label == prediction:
                correct_count += 1
                break

    return (correct_count / total) * 100

if __name__ == '__main__':
    ## Lemmas are returned in key, value where key is lemma_id and value is WSDInstance (lemma_id, sentence_id, lemma, context, index)
    semcor_lemmas = load_instances(SEMCOR_DATA_FILE)
    semcor_labels = get_labels(SEMCOR_LABELLED)

    senseval_2_lemmas = load_instances(SENSEVAL_2_DATA_FILE)
    senseval_2_labels = get_labels(SENSEVAL_2_LABELLED)

    senseval_3_lemmas = load_instances(SENSEVAL_3_DATA_FILE)
    senseval_3_labels = get_labels(SENSEVAL_3_LABELLED)

    ## Driver code for distributional lesk
    ## Select 5000 sentences
    semcor_5000_lemmas = get_semcor_data(semcor_lemmas)
    mapping_dict = read_mapping()
    lexeme_embeds = read_lexeme_embeds()
    synset_embeds = read_synset_embeds()

    ## The word embeddings are updated with the train data -- 5000 semcor sentences
    train_dist_lesk(semcor_5000_lemmas, mapping_dict, lexeme_embeds, synset_embeds)

    all_embeddings = np.array(word_embeddings)
    np.save('dist_word_embeddings.npy', all_embeddings)

    ## Driver code for evaluation of distributional lesk
    sem_accuracy = eval_dist_lesk(semcor_lemmas, semcor_labels, mapping_dict, lexeme_embeds)
    print("Dist_lesk accuracy on Semcor:", sem_accuracy)

    #senseval_2_accuracy = eval_dist_lesk(senseval_2_lemmas, senseval_2_labels, mapping_dict, lexeme_embeds)
    #print("Dist_lesk accuracy on Senseval_2:", senseval_2_accuracy)

    #senseval_3_accuracy = eval_dist_lesk(senseval_3_lemmas, senseval_3_labels, mapping_dict, lexeme_embeds)
    #print("Dist_lesk accuracy on Senseval_3:", senseval_3_accuracy)