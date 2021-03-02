"""
Module with functions that could be helpful to create dataloaders.

TODO: unite old and new create_dataloader functions.

"""


from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from allennlp.modules.elmo import Elmo, batch_to_ids

import numpy as np
import torch


def tokenize_and_preserve_labels(bert_tokenizer, sentence, all_labels):
    """
    Tokenizes sentence using given pretrained BERT tokenizer.

    param labels_lists should be [ list of [list of [list with labels] for every sample] for every head ]
    
    """

    for sl in all_labels:
      if len(sentence) != len(sl):
        raise ValueError("Not the same length of sentence and labels sets")
      
    tokenized_sentence = []
    new_all_labels = [[] for _ in range(len(all_labels))]

    for i in range(len(sentence)):
        word = sentence[i]
        word_labels = [ls[i] for ls in all_labels]
        
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = bert_tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Extend all versions of sentence labels
        for j in range(len(all_labels)):
            new_all_labels[j].extend([word_labels[j]] * n_subwords)

    tokenized_sentence = ['[CLS]'] + tokenized_sentence + ['[SEP]']
    new_all_labels = [['O'] + labels + ['O'] for labels in new_all_labels]

    return tokenized_sentence, new_all_labels
    

def create_dataloader(conll_obj, tag_names, bert_tokenizer, datatype='train', desired_pad='max', batch_size=128):
    """
    Creates dataloader for our model with multiple heads.

    Parameters
    ----------
    
    conll_obj: object
      instance of conll class (whith multiple heads)
    tag_names: list of str
      names of tags should be from keys of conll.conll.one_tag2idx
    bert_tokenizer:
    datatype: str, default='train'
      'train'/'valid'/'test'
    desired_pad: str or int, default='max'
      'max'/'mean'/int
    batch_size: int, default=128

    Returns
    -------
    TensorDataset, RandomSampler (for valid and test SequentialSampler), DataLoader

    TensorDataset consist of elements: 
        (bert ids of tokens in sentence, elmo vectors of tokens in sentence, all labels of sentence, mask)
        where `all labels of sentence` - tensor with shape (number of heads, padded len)

    """
    
    
    num_of_heads = len(tag_names)
    
    _all_labels = [conll_obj.one_tag_dict[datatype][name] for name in tag_names]
    
    data_tokenized = [tokenize_and_preserve_labels(bert_tokenizer, item[0], item[1:]) for item in zip(conll_obj.sentences[datatype], *_all_labels)]
    data_tokens = [x[0] for x in data_tokenized]
    data_labels = [[x[1][i] for x in data_tokenized] for i in range(num_of_heads)]

    if desired_pad=='max':
        DISIRED_LENGTH = np.max([len(sen) for sen in data_tokens])
    elif desired_pad=='mean':
        DISIRED_LENGTH = int(np.mean([len(sen) for sen in data_tokens]))
    elif isinstance(desired_pad, int):
        DISIRED_LENGTH = desired_pad
    else:
        raise ValueError("How it should be padded?")
        
    data_ids = pad_sequences([bert_tokenizer.convert_tokens_to_ids(txt) for txt in data_tokens],
                          maxlen=DISIRED_LENGTH, dtype="long", value=0.0,
                          truncating="post", padding="post")
  
    data_tags = [pad_sequences([[conll_obj.one_tag2idx[name].get(l) for l in seq_labels] for seq_labels in data_labels[i]],
                      maxlen=DISIRED_LENGTH, value=conll_obj.one_tag2idx[name]["PAD"], padding="post",
                      dtype="long", truncating="post") for i, name in enumerate(tag_names)]
  
    data_masks = [[float(i != 0.0) for i in ii] for ii in data_ids]
    
    # Creating tensors
    data_elmo_ids = batch_to_ids(data_tokens)
    data_bert_ids = torch.tensor(data_ids)

    # We need to pad elmo ids with zeros to have the same sequence length as BERT.
    if data_elmo_ids.shape[1] < data_bert_ids.shape[1]:
        data_elmo_ids = torch.cat((data_elmo_ids,
                                    torch.zeros((data_elmo_ids.shape[0],
                                                data_bert_ids.shape[1]-data_elmo_ids.shape[1],
                                                data_elmo_ids.shape[2]))), dim=1).type(torch.LongTensor)
    data_tags = torch.tensor(data_tags)

    if data_tags.shape[0] == 1: # in case of one head remove unnecessary dimension
        data_tags = data_tags.view(data_tags.shape[1], data_tags.shape[2])
    
    if len(data_tags.shape) == 3:
        data_tags = data_tags.transpose(1,0) # to have shape (batch, head, seq)
    
    data_masks = torch.tensor(data_masks)
                              
    data_dataset = TensorDataset(data_bert_ids, data_elmo_ids, data_tags, data_masks)
    if datatype == 'train':
        data_sampler = RandomSampler(data_dataset)
    else:
        data_sampler = SequentialSampler(data_dataset)
    data_dataloader = DataLoader(data_dataset, sampler=data_sampler, batch_size=batch_size)

    return data_dataset, data_sampler, data_dataloader
    

def union_labels(all_labels, tag_names, idx2one_tag=None, keep=None):
    """
    Unions sequences of labels in one sequence (it's useful for multiple heads case)
    We suppose that one token could has just one "special" tag.
    
    Parameters
    ----------
    all_labels: [list of [list of [list of tags] for samples] for heads]
    
    tag_names: list of str
        names of tags should be from keys of conll.conll.idx2one_tag
    
    idx2one_tag: dict, default=None
        if passed labels are indexes of tags - this dict should be given (conll.idx2one_tag)
    
    keep: str, default=None
        Which tag to keep if there are several possibilities for one token
        It can be 'first', 'last' or 'both'
        If None - error will be raised
    
    """
    
    num_of_heads = len(all_labels)
    num_of_samples = len(all_labels[0])
    
    result = []
    
    for i in range(num_of_samples):
        seq_len = len(all_labels[0][i])
        sample_tags = []
        for j in range(seq_len):
            possible_tags = [all_labels[h][i][j] for h in range(num_of_heads)]
            possible_tags = [t if isinstance(t, str) else idx2one_tag[tag_names[k]][t] for k, t in enumerate(possible_tags)]
            next_tag = possible_tags[0]
            if next_tag == 'PAD': #all next tags are paddings
                sample_tags.extend(['PAD']*(seq_len-j))
                break
            for t in possible_tags[1:]:
                if t != 'O' and t != 'PAD':
                    if next_tag != 'O' and next_tag != 'PAD':
                        if keep is None:
                            raise ValueError("One token should has one tag")
                        elif keep == 'first':
                            continue
                        elif keep == 'last':
                            next_tag = t
                        else:
                            if isinstance(next_tag, str):
                                next_tag = [next_tag, t]
                            else:
                                next_tag.append(t)
                    else:
                        next_tag = t
            sample_tags.append(next_tag)
        result.append(sample_tags)
    
    return result
    

def create_dataloader_old(data, labels, tag2idx, tokenizer, datatype='train', desired_pad='max', batch_size=128):
  """
  Creates dataloader for one-head case.

  returns: TensorDataset, RandomSampler (for valid and test SequentialSampler), DataLoader
  """
  data_tokenized = [tokenize_and_preserve_labels(tokenizer, s, [l]) for s, l in zip(data, labels)]
  data_tokens = [x[0] for x in data_tokenized]
  data_labels = [x[1][0] for x in data_tokenized]

  if desired_pad=='max':
    DISIRED_LENGTH = np.max([len(sen) for sen in data_tokens])
  elif desired_pad=='mean':
    DISIRED_LENGTH = int(np.mean([len(sen) for sen in data_tokens]))
  elif isinstance(desired_pad, int):
    DISIRED_LENGTH = desired_pad
  else:
    raise ValueError("How it should be padded?")
  
  data_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in data_tokens],
                          maxlen=DISIRED_LENGTH, dtype="long", value=0.0,
                          truncating="post", padding="post")
  
  data_tags = pad_sequences([[tag2idx[l] for l in seq_labels] for seq_labels in data_labels],
                     maxlen=DISIRED_LENGTH, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")
  
  data_masks = [[float(i != 0.0) for i in ii] for ii in data_ids]

  # Creating tensors
  data_elmo_ids = batch_to_ids(data_tokens)
  data_bert_ids = torch.tensor(data_ids)
  data_tags = torch.tensor(data_tags)
  data_masks = torch.tensor(data_masks)

  # We need to pad elmo ids to have the same sequence length as BERT.
  if data_elmo_ids.shape[1] < data_bert_ids.shape[1]:
    data_elmo_ids = torch.cat((data_elmo_ids,
                                torch.zeros((data_elmo_ids.shape[0],
                                             data_bert_ids.shape[1]-data_elmo_ids.shape[1],
                                             data_elmo_ids.shape[2]))), dim=1).type(torch.LongTensor)
    
  data_dataset = TensorDataset(data_bert_ids, data_elmo_ids, data_tags, data_masks)
  if datatype == 'train':
    data_sampler = RandomSampler(data_dataset)
  else:
    data_sampler = SequentialSampler(data_dataset)
  data_dataloader = DataLoader(data_dataset, sampler=data_sampler, batch_size=batch_size)

  return data_dataset, data_sampler, data_dataloader

    