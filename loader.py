import xml.etree.cElementTree as ET
import codecs


class WSDInstance:
    def __init__(self, lemma_id, sent_id, lemma, context, index):
        self.lemma_id = lemma_id  # id of the WSD instance
        self.sent_id = sent_id   # id of the sentence in which lemma occurs, helps to select 5000 sentences
        self.lemma = lemma  # lemma of the word whose sense is to be resolved
        self.context = context  # lemma of all the words in the sentence context
        self.index = index  # index of lemma within the sentence

    def __str__(self):
        '''
        For printing purposes.
        '''
        return '%s\t%s\t%s\t%d' % (self.lemma_id, self.sent_id, self.lemma, ' '.join(self.context), self.index)


def load_instances(f):
    '''
    Load two lists of cases to perform WSD on. The structure that is returned is a dict, where
    the keys are the ids, and the values are instances of WSDInstance.
    '''
    tree = ET.parse(f)
    root = tree.getroot()

    instances = {}

    for text in root:
        for sentence in text:
            # construct sentence context
            context = [(el.attrib['lemma']) for el in sentence]
            for ind, el in enumerate(sentence):
                if el.tag == 'instance':
                    lemma_id = el.attrib['id']
                    sent_id = el.attrib['id'][0:9]
                    lemma = (el.attrib['lemma'])
                    instances[lemma_id] = WSDInstance(lemma_id, sent_id, lemma, context, ind)
    return instances

def get_labels(LABEL_FILE):
  """
  Reads the labels/annotations of the lemmas and returns in dictionary form
  """
  labels = {}
  for line in open(LABEL_FILE):
    if len(line) <= 1: continue
    lemma_id_label = line.strip().split(" ")

    labels[lemma_id_label[0]] = lemma_id_label[1:]

  return labels