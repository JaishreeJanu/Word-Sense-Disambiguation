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


def sort_lemmas(lemmas):
    """
    Sort lemmas based on their number of senses
    :param lemmas:
    :return:
    """
    print("SORTING THE LEMMAS BASED ON THEIR NUMBER OF SYNSETS")
    sorted_lemmas = dict(sorted(lemmas.items(), key=lambda item: item[1].no_synsets))
    return sorted_lemmas


def eval_acc(pred_synset, correct_label):
    """
    Checks if the prediction matches with the correct label
    """
    correct_bool = False
    try:
        for synset_lemma in pred_synset.lemmas():
            if synset_lemma.key() == correct_label:
                correct_bool = True
                break
    except:
        print("The keys for the pred_synset could not be found !!!!!!!!!!!!!!!!!")

    return correct_bool