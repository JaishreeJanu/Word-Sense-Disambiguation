### Fine tuning sbert and evluating distributional lesk on sbert sentence embeddings
from loader import *
from dict_utilities import *

from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn

SEMCOR_DATA_FILE = 'WSD_Unified_Evaluation_Datasets/SemCor/semcor.data.xml'
SEMCOR_LABELLED = 'WSD_Unified_Evaluation_Datasets/SemCor/semcor.gold.key.txt'


if __name__ == '__main__':
    semcor_lemmas = load_instances(SEMCOR_DATA_FILE)
    semcor_labels = get_labels(SEMCOR_LABELLED)

    # Define the model. Either from scratch of by loading a pre-trained model
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    ## Fine tune sbert on 5000 of semcor instances
    train_examples = []

    semcor_5000_lemmas = get_semcor_data(semcor_lemmas)
    for lemma_id, wsd_inst in semcor_5000_lemmas.items():
        label_key = semcor_labels[lemma_id][0]
        lemma = wsd_inst.lemma
        context = wsd_inst.context
        gloss = wn.lemma_from_key(label_key).synset().definition()

        for synset in wn.synsets(lemma):
            for synset_lemma in synset.lemmas():
                this_synset_key = synset_lemma.key()
                if this_synset_key != label_key:
                    gloss_opposite = synset.definition()
            this_synset_key = this_synset_key[:-1]  # Remove the last comma

        train_examples += [InputExample(texts=[context, gloss], label=0.95),
                          InputExample(texts=[context, gloss_opposite], label=0.25)]

    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

    model.save(path="sbert_model/", model_name='semcor_finetuned')

    ## Test the model
    query_embedding = model.encode('This framework generates embeddings for each input sentence')
    passage_embedding = model.encode(['Sentences are passed as a list of string.',
                                      'The quick brown fox jumps over the lazy dog.'])

    print("Similarity: ", util.dot_score(query_embedding, passage_embedding))


