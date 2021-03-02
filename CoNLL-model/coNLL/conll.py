import json

class CoNLL_old():
    """
    For one head case
    """
    def __init__(self, path):
        """
        Loads data from filesystem
        """
        self.data = {} # raw data
        self.sentences = {} # split data
        self.labels = {} # split labels
        self.types = ['train', 'valid', 'test']
        
        
        for typ in self.types:
            with open(path+typ+'.txt', 'r') as f:
                self.data[typ] = f.read()
        
    
    def split_text_label(self, typ):
        """
        Splits text (train/valid/test) to sentences (lists of words) and their labels (lists of labels).
        """
        
        split_labeled_text = []
        sentence = []
        for line in self.data[typ].split('\n'):
            if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
                if len(sentence) > 0:
                    split_labeled_text.append(sentence)
                    sentence = []
                continue
            splits = line.split(' ')
            sentence.append([splits[0],splits[-1].rstrip("\n")])
        if len(sentence) > 0:
            split_labeled_text.append(sentence)
            sentence = []
        self.sentences[typ] = []
        self.labels[typ] = []
        for sent in split_labeled_text:
            sentence = []
            label = []
            for s_l in sent:
                sentence.append(s_l[0])
                label.append(s_l[1])
            self.sentences[typ].append(sentence)
            self.labels[typ].append(label)
    
    
    def create_tag2idx(self, path):
        """
        Loads fixed tag2idx dict if possible or creates new.
        """
        
        try:
          with open(path, 'r') as f:
            self.tag2idx = json.load(f)
        except:
          tag_values = set()
          for l in self.labels['train']:
              tag_values.update(l)
          tag_values.update(["PAD"])
          self.tag2idx = {t: i for i, t in enumerate(tag_values)}
          with open(path, 'w') as f:
            json.dump(self.tag2idx,f)
    
    def create_idx2tag(self):
        """
        Should be used after create_tag2idx
        """
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}
    
    
        
        

class CoNLL(CoNLL_old):
    """
    For multiple head case
    """
    def __init__(self, path):
        super(CoNLL, self).__init__(path)
        self.one_tag_dict = {} # for multiple heads
    
    def create_set_of_labels(self):
        """
        Creates a set of all labels in the train sentences. 
        
        split_text_labels('train') should be run first
        
        """
        
        self.set_of_labels = set()
        for sen in self.labels['train']:
            self.set_of_labels.update(sen)
        
    def create_one_labeled_data(self, typ):
        """
        Creates a set of labels for sentences leaving just one "special" and 'O' (outside) token.
        We need it for model with more than 1 CRF heads.
        P.S. It considers, for example, B-LOC and I-LOC labels as one.
        """
        
        self.one_tag_dict[typ] = {}
        
        for tag in self.set_of_labels:
            if tag == 'O' or tag == 'PAD':
                continue
            tag_without_prefix = tag[2:] #B-LOC -> LOC
            tag_map = lambda tag: tag if tag_without_prefix in tag else ('PAD' if tag=='PAD' else 'O')
            one_tag_labels = [[tag_map(tag) for tag in sen]for sen in self.labels[typ]]
            self.one_tag_dict[typ][tag_without_prefix] = one_tag_labels
    
    def create_one_tag2idx(self, path):
        """
        Create one_tag2idx attribute with keys like 'LOC', 'ORG' and etc.
        So, each head will have custom mapping from its labels to their idx 
        (we need it because CRF requires idxes to be in range(len(head_labels)))
        
        
        """
        
        try:
          with open(path, 'r') as f:
            self.one_tag2idx = json.load(f)
        except:
          self.one_tag2idx = {}
          for k in self.one_tag_dict['train'].keys():
              _set_of_labels = set()
              self.one_tag2idx[k] = {}
              for sen in self.one_tag_dict['train'][k]:
                  _set_of_labels.update(sen)
              _set_of_labels.update(['PAD'])
              for i, l in enumerate(_set_of_labels):
                  self.one_tag2idx[k][l] = i
          with open(path, 'w') as f:
            json.dump(self.one_tag2idx,f)
        
    def create_idx2one_tag(self):
        """
        Should be used after create_one_tag2idx
        
        """
        
        self.idx2one_tag = {}
        for t in self.one_tag2idx.keys():
            self.idx2one_tag[t] = {v: k for k, v in self.one_tag2idx[t].items()}
        
    