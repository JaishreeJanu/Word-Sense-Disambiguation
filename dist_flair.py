"""
Implementation of distributional lesk with flair embeddings
"""
#import nltk
##nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn

from numpy.linalg import norm
import pickle as pk

from loader import *
from helper import *
from dict_utilities import *

from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from sklearn.decomposition import PCA
from flair.data import Sentence
import numpy as np
from tqdm import tqdm

SEMCOR_DATA_FILE = 'SemCor/semcor.data.xml'
SEMCOR_LABELLED = 'SemCor/semcor.gold.key.txt'
SENSEVAL_2_DATA_FILE = './senseval2/senseval2.data.xml'
SENSEVAL_2_LABELLED = './senseval2/senseval2.gold.key.txt'
SENSEVAL_3_DATA_FILE = './senseval3/senseval3.data.xml'
SENSEVAL_3_LABELLED = './senseval3/senseval3.gold.key.txt'

## Just test sentences
context = 'Influential people are more important.'
gloss = 'having authority or ascendancy or influence'
gloss_opposite = 'of extreme importance; vital to the resolution of a crisis'

def def_flair_embed():
    """
    Initializes the stack of flair embeddings
    """
    # init Flair forward and backwards embeddings
    flair_embedding_forward = FlairEmbeddings('news-forward')
    flair_embedding_backward = FlairEmbeddings('news-backward')

    # create a StackedEmbedding object that combines glove and forward/backward flair embeddings
    stacked_embeddings = StackedEmbeddings([
        flair_embedding_forward,
        flair_embedding_backward,
    ])
    return stacked_embeddings

def get_flair_embed(lst_strings, pca_trained):
    """
    Returns the flair embedding for the sentence
    """
    context_sentence = Sentence(' '.join(lst_strings))
    stacked_embeddings = def_flair_embed()
    stacked_embeddings.embed(context_sentence)

    flair_embeds = []
    for token in context_sentence:
        flair_embeds.append(token.embedding.numpy())
    flair_embeds = pca_trained.transform(flair_embeds)

    ## Find mean of the embeddings
    flair_embeds = np.mean(flair_embeds, axis=0)

    return flair_embeds

def train_pca(senseval_2_lemmas, new_dim = 300):
    """
    Train PCA model to transform flair embeddings to 300-dimensional
    """
    senseval_2_lemmas = dict(list(senseval_2_lemmas.items())[:1000]) ## Only selected 400 examples

    stacked_embeddings = def_flair_embed()
    ## Get the training examples for PCA
    pca_train_embeds = []
    for lemma_id, wsd in tqdm(senseval_2_lemmas.items()):
        context_sentence = Sentence(' '.join(wsd.context))
        stacked_embeddings.embed(context_sentence)

        for token in context_sentence:
            pca_train_embeds.append(token.embedding.numpy())

    pca_train_embeds = np.asarray(pca_train_embeds)
    ## train PCA for mapping 4096-dim to 300-dim componenets, since we have lexeme embeds in 300-dim
    pca = PCA(n_components=new_dim)
    pca.fit(pca_train_embeds)

    return pca

def eval_dist_flair(lemmas, labels, mapping_dict, lexeme_embeds, pca_trained):
    """
    Evaluate distributional lesk with flair embeddings
    """
    correct_count = 0
    total = len(labels)

    for lemma_id, lemma_label in tqdm(dict(list(labels.items())[:500]).items()):
        this_wsd_inst = lemmas[lemma_id]

        ## Get the context embeds
        context_embed = get_flair_embed(this_wsd_inst.context, pca_trained)
        final_score = 0
        final_synset = this_wsd_inst.lemma[0]
        final_wn_synset_id = ''
        for this_synset in wn.synsets(this_wsd_inst.lemma):
            ## Computations for this synset, lemma pair
            ## Get the gloss embeds
            gloss_embed = get_flair_embed(this_synset.definition().split(" "), pca_trained)
            wn_synset_id = ''
            for synset_lemma in this_synset.lemmas():
                this_synset_key = synset_lemma.key()
                if this_synset_key in mapping_dict.keys():
                    wn_synset_id = mapping_dict[this_synset_key]
                    break

            ## Get the wn-id from mapping.txt and using above dictionary
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

        ## Call the eval function for finding accuracy
        correct_label = labels[lemma_id][0]
        if eval_acc(final_synset, correct_label):
            correct_count += 1

    return (correct_count / total) * 100

if __name__ == '__main__':
    semcor_lemmas = sort_lemmas(load_instances(SEMCOR_DATA_FILE))
    semcor_labels = get_labels(SEMCOR_LABELLED)

    senseval_2_lemmas = sort_lemmas(load_instances(SENSEVAL_2_DATA_FILE))
    senseval_2_labels = get_labels(SENSEVAL_2_LABELLED)

    senseval_3_lemmas = sort_lemmas(load_instances(SENSEVAL_3_DATA_FILE))
    senseval_3_labels = get_labels(SENSEVAL_3_LABELLED)

    mapping_dict = read_mapping()
    lexeme_embeds = read_lexeme_embeds()

    #pca_trained = train_pca(senseval_2_lemmas)
    #pk.dump(pca_trained, open("pca_trained.pkl", "wb"))
    pca_trained = pk.load(open("pca_trained.pkl", 'rb'))

    #semcor_labels = dict(list(semcor_labels.items())[:2500])
    #semcor_2_acc = eval_dist_flair(semcor_lemmas, semcor_labels, mapping_dict, lexeme_embeds, pca_trained)
    #print("Accuracy:: ", semcor_2_acc)

    #senseval_2_acc = eval_dist_flair(senseval_2_lemmas, senseval_2_labels, mapping_dict, lexeme_embeds, pca_trained)
    #print("Accuracy:: ", senseval_2_acc)

    senseval_3_acc = eval_dist_flair(senseval_3_lemmas, senseval_3_labels, mapping_dict, lexeme_embeds, pca_trained)
    print("Accuracy:: ", senseval_3_acc)