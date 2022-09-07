#### Helper functions for reading mapping, lexeme and synset text files
import numpy as np

def read_mapping():
    '''
    ## Read the mapping file and store it into dictionary
    ##  lemma_sense_id lemma%sense_id, lemma%sense_id
    :return: mapping_dict
    '''
    """
      Reads the mapping.txt file which contains mapping from wn-ids to sense keys
      Returns these mappings in dictionary format
      """
    mapping_dict = {}
    for line in open(file = 'embeddings/mapping.txt'):
        if len(line.strip().split(' ')) < 2: continue
        wn_synset_id, synsets_keys = line.strip().split(' ')
        all_synset_keys = synsets_keys.split(",")

        for synset_key in all_synset_keys:
            if synset_key != '':
                mapping_dict[synset_key] = wn_synset_id

    return mapping_dict

def read_lexeme_embeds(file='./embeddings/lexemes.txt'):
  """
  Reads the lexeme embeds corresponding to each wn_id
  Returns the dictionary
  """

  lexeme_dict = {}
  for line in open(file):
    if len(line.strip().split(' ')) < 2: continue

    wn_synset_id_lexeme_embed = line.strip().split(' ')
    wn_synset_id = wn_synset_id_lexeme_embed[0][wn_synset_id_lexeme_embed[0].find('-')+1:]
    lexeme_dict[wn_synset_id] = np.asarray(np.array(wn_synset_id_lexeme_embed[1:]), dtype=float)

  return lexeme_dict

def read_synset_embeds(file='./embeddings/synsets.txt'):
  """
  Reads the lexeme embeds corresponding to each wn_id
  Returns the dictionary
  """

  synset_dict = {}
  for line in open(file):
    if len(line.strip().split(' ')) < 2: continue

    wn_synset_id_synset_embed = line.strip().split(' ')
    synset_dict[wn_synset_id_synset_embed[0]] = np.asarray(np.array(wn_synset_id_synset_embed[1:]), dtype=float)

  return synset_dict

