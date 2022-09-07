### Fine tuning sbert and evluating distributional lesk on sbert sentence embeddings
from loader import *
from dict_utilities import *
from helper import *

import gzip
import csv
import random
import torch
import os
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, InputExample, losses, util, models
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
from numpy.linalg import norm

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


def train_pca(new_dimension):
    """
    We want to get a PCA matrix so as to reduce the dimensions of sbert embeddings
    It returns the sbert model with pca added in dense layer
    """
    ## Get the sentences to compute PCA
    nli_dataset_path = 'datasets/AllNLI.tsv.gz'
    if not os.path.exists(nli_dataset_path):
        util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

    ## Define the model
    #word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
    #pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    ##model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Read sentences from NLI dataset
    nli_sentences = set()
    with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            nli_sentences.add(row['sentence1'])
            nli_sentences.add(row['sentence2'])

    nli_sentences = list(nli_sentences)
    random.shuffle(nli_sentences)

    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    # To determine the PCA matrix, we need some example sentence embeddings.
    # Here, we compute the embeddings for 20k random sentences from the AllNLI dataset
    pca_train_sentences = nli_sentences[0:20000]
    train_embeddings = model.encode(pca_train_sentences, convert_to_numpy=True)

    # Compute PCA on the train embeddings matrix
    pca = PCA(n_components=new_dimension)
    pca.fit(train_embeddings)
    pca_comp = np.asarray(pca.components_)

    # Define the model. Either from scratch of by loading a pre-trained model

    # We add a dense layer to the model, so that it will produce directly embeddings with the new size
    dense = models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=new_dimension, bias=False,
                         activation_function=torch.nn.Identity())
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
    model.add_module('dense', dense)

    return model

def train_sbert(lemmas, labels, model):
    """
    Fine tune the sbert model on semcor/senseval instances
    """
    train_examples = []

    for lemma_id, wsd_inst in lemmas.items():
        label_key = labels[lemma_id][0]
        lemma = wsd_inst.lemma
        context = wsd_inst.context

        for synset in wn.synsets(lemma):
            if wn.lemma_from_key(label_key).synset() == synset:
                correct_gloss = synset.definition()
                train_examples += [InputExample(texts=[context, correct_gloss], label=0.95)]
            else:
                incorrect_gloss = synset.definition()
                train_examples += [InputExample(texts=[context, incorrect_gloss], label=0.15)]

    print(len(train_examples))
    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    ## Cosine similarity is used as the loss function
    train_loss = losses.CosineSimilarityLoss(model)

    # Fine-tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
    # Save the model
    model.save(path="bert_multi_mini/")

    return model

def dist_sbert(lemmas, labels, model, mapping_dict, lexeme_embeds):
    """
    Evaluation function for distributional lesk with sbert embeddings
    """
    correct_count = 0
    total = len(labels)

    for lemma_id, lemma_label in tqdm(labels.items()):
        this_wsd_inst = lemmas[lemma_id]

        ## Get the context embeds
        context_embed = model.encode(' '.join(this_wsd_inst.context)) ## Since context is in list of strings format, join it to make a single string of a sentence
        final_score = 0
        final_synset = this_wsd_inst.lemma[0]
        final_wn_synset_id = ''
        for this_synset in wn.synsets(this_wsd_inst.lemma):
            ## Computations for this synset, lemma pair
            ## Get the gloss embeds
            gloss_embed = model.encode(this_synset.definition())
            ## This key is formed to get the lexeme embedding
            wn_synset_id = ''
            for synset_lemma in this_synset.lemmas():
                this_synset_key = synset_lemma.key()
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
                if score > final_score:
                    final_score = score
                    final_wn_synset_id = wn_synset_id
                    final_synset = this_synset


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

    semcor_5000_lemmas = sort_lemmas(get_semcor_data(semcor_lemmas))
    ## Get the PCA trained model
    model = train_pca(new_dimension=300) ## SInce the lexeme embeddings are 300-dimensional
    ## Train the model
    model = train_sbert(senseval_2_lemmas, senseval_2_labels, model)


    ## Load the saved model
    model = SentenceTransformer('./bert_multi_mini/')

    ## Test the model
    query_embedding = model.encode(context)
    passage_embedding = model.encode([gloss, gloss_opposite])
    print("Similarity: ", util.dot_score(query_embedding, passage_embedding))

    ## Driver code for evaluating dist lesk with sbert embeddings
    mapping_dict = read_mapping()
    lexeme_embeds = read_lexeme_embeds()
    synset_embeds = read_synset_embeds()

    semcor_labels = dict(list(semcor_labels.items())[:3000])
    sem_accuracy = dist_sbert(semcor_lemmas, semcor_labels, model, mapping_dict, lexeme_embeds)
    print("Dist_lesk accuracy on Semcor:", sem_accuracy)

    senseval_2_accuracy = dist_sbert(senseval_2_lemmas, senseval_2_labels, model, mapping_dict, lexeme_embeds)
    print("Dist_lesk accuracy on Senseval_2:", senseval_2_accuracy)

    senseval_3_accuracy = dist_sbert(senseval_3_lemmas, senseval_3_labels, model, mapping_dict, lexeme_embeds)
    print("Dist_lesk accuracy on Senseval_3:", senseval_3_accuracy)


