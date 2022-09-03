## Helper functions for creating dictionary variables to store senses, gloss vector and context vector
import numpy as np
from nltk.corpus import wordnet as wn

def get_semcor_data(lemmas):
    '''
    Reads Semcor data and selects first 5000 sentences from it
    :return: returns dictionary of 5000 sentences in form key: lemma_id, value:  WSDInstance
    '''

    ## Inefficient code: for loops !!!
    selected_lemmas = {}
    sent_count = 1
    curr_sent = lemmas[list(lemmas.keys())[0]].sent_id  ## Pick the sentence id for the first sentence
    for key, value in lemmas.items():
        selected_lemmas[key] = value
        if value.sent_id == curr_sent:
            continue
        else:
            curr_sent = value.sent_id
            sent_count += 1

        if sent_count >= 5000:
            break

    return selected_lemmas

def get_word_embed(selected_lemmas, word_embeddings):
    '''
    ## word_embed_dict = {}, key: lemma_id, value: word_embedding
    :return:
    '''
    word_embed_dict = {}
    for key, value in selected_lemmas.items():
        ## Since, word_embeddings not found for some words
        this_lemma = value.lemma
        try:
            word_embed_dict[key] = word_embeddings[this_lemma]
        except:
            '''
            Word embeddings not found for two-word phrases like: "for example", "go up", "blue collar", "coffee break", etc.
            So, split these words and store their embeddings separately
            '''
            if len(this_lemma.split('-')) == 2:
                lemma_1, lemma_2 = this_lemma.split('-')
                try:
                    word_embed_dict[key] = word_embeddings[lemma_1]
                    word_embed_dict[key] = word_embeddings[lemma_1]
                except:
                    pass

            elif len(this_lemma.split('_')) == 2:
                lemma_1, lemma_2 = this_lemma.split('_')
                try:
                    word_embed_dict[key] = word_embeddings[lemma_1]
                    word_embed_dict[key] = word_embeddings[lemma_1]
                except:
                    pass

            else:
                # print("Embedding cound not be found for: ", this_lemma)
                pass

    return word_embed_dict