# Bert pre-trained model selected in the list: bert-base-uncased, 
# bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.
BERT_MODEL = 'bert-base-cased'

# The name of the task to train.I'm going to name this 'yelp'.
TASK_NAME = 'first'

# The output directory where the fine-tuned model and checkpoints will be written.
OUTPUT_DIR = f'outputs/{TASK_NAME}/'

# The directory where the evaluation reports will be written to.
REPORTS_DIR = f'reports/{TASK_NAME}_evaluation_report/'


# This is where BERT will look for pre-trained models to load parameters from.
CACHE_DIR = 'cache/'

# create BERT model
#bert = BertForTokenClassification.from_pretrained(
#                        BERT_MODEL,
#                        output_hidden_states=True)
#        
#for pars in bert.parameters():
#    pars.requires_grad = False
