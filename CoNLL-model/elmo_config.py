# medium ELMo weights
weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5'

# medium ELMO options
options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json'

# create ElMO model (we've already found that to use 2 elmo layers is the best choise)
#elmo = Elmo(options_file, weight_file, num_output_representations=2,
#                  dropout=0, requires_grad=False)

#elmo(train_elmo_ids[:2])['elmo_representations'][0].shape