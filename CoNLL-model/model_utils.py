"""
Some useful model utils.
"""

import tqdm
import json

import torch

from transformers import BertTokenizer
from seqeval.metrics import f1_score, accuracy_score


def eval_model(model, dataloader, device, conll):
    """
    Evaluates BEbic model using given dataloader.

    Returns: tuple - (accuracy, f1_score)
    
    """
    model.eval()

    mean_loss, mean_acc, mean_f1 = 0, 0, 0
    head_predictions = {head: [] for head in model.heads.keys()}
    head_true_labels = {head: [] for head in model.heads.keys()}
    epoch_head_loss = {head: 0 for head in model.heads.keys()}
    head_result = {head: {} for head in model.heads.keys()}

    for batch in dataloader:
        if device.type != 'cpu':
            batch = tuple(t.to(device) for t in batch)
        b_bert_ids, b_elmo_ids, b_labels, b_input_mask = batch

        with torch.no_grad():
            # forward pass of shared layers
            bilstm_logits = model.shared_forward(b_bert_ids, b_elmo_ids, b_input_mask.byte())
            
            #loss = 0
            #predicted_tags = []
            for i, head in enumerate(model.heads.keys()):
                # forward pass for every head separately
                _b_head_labels = b_labels[:,i,:] if len(model.heads.keys()) > 1 else b_labels
                epoch_head_loss[head] += model.get_one_head_loss(bilstm_logits, 
                                                                _b_head_labels, 
                                                                b_input_mask, 
                                                                head).item()
                # predicted indexes
                _pred_idxes = model.get_one_head_seq(bilstm_logits,
                                                    b_input_mask,
                                                    head)
                # predicted idxes -> tags
                map_dict = conll.idx2one_tag[head]
                for pi in _pred_idxes:
                    _pred_labels = [map_dict[x] for x in pi]
                    head_predictions[head].append(_pred_labels)

        # move loss to cpu
        labels_ = b_labels.detach().cpu()
        #_true_labels_with_pads = union_labels(labels_.transpose(0,1).tolist(), TAG_NAMES, conll.idx2one_tag)

        # true idxes -> tags, mean loss evaluation
        for i, head in enumerate(model.heads.keys()):
            # mean loss over batch
            epoch_head_loss[head] /= len(dataloader)
            mean_loss += epoch_head_loss[head]

            _true_labels_with_pads = labels_[:,i,:].tolist() if len(model.heads.keys()) > 1 else labels_.tolist()
            
            map_dict = conll.idx2one_tag[head]
            for tl in _true_labels_with_pads:
                _true_labels = [map_dict[x] for x in tl if map_dict[x] != 'PAD']
                head_true_labels[head].append(_true_labels)

    for head in model.heads.keys():
        acc = accuracy_score(head_predictions[head],head_true_labels[head])
        f1 = f1_score(head_predictions[head],head_true_labels[head])
        head_result[head]['acc'] = acc
        head_result[head]['f1'] = f1
        mean_acc += acc/model.num_heads
        mean_f1 += f1/model.num_heads
    return head_result, mean_loss, mean_acc, mean_f1


def train(model, train_dataloader, optimizer, device, conll,
          scheduler=None, n_epoch=5,
          max_grad_norm=None, valid_dataloader=None, show_info=True, 
          save_model=True, path_to_save=None, ver='v1'):
    """
    Trains BEbic model.

    Parameters:
    -----------
    model:
    train_dataloader:
    optimizer:
    device:
    scheduler: default=None
    n_epoch: int, default=5
    max_grad_norm: default=None
    valid_dataloader: default=None
    show_info: bool, default=True
      Whether to print messages.
    save_model: bool, default=True
      Whether to save model
    path_to_save: str, default=None
      Where to save model
    ver: str, default='v1'
      Model's version

    Return:
    -------
    loss_value: list of average head train losses 
    head_results: dict with information of validation results for every head.
    Returns in case when validation has been provided.

    """
    loss_values = []
    if valid_dataloader is not None:
        head_results = {head: {'losses': [], 'accs': [], 'f1': []} for head in model.heads.keys()}

    for e in range(n_epoch):
        if show_info:
          print(f"\nEpoch #{e}")
        ########## Training ###########

        model.train()

        total_loss = 0

        if show_info:
            enumerator = enumerate(tqdm.tqdm(train_dataloader, position=0, leave=True))
        else:
            enumerator = enumerate(train_dataloader)

        for step, batch in enumerator:
            if device.type != 'cpu':
                batch = tuple(t.to(device) for t in batch)
            b_bert_ids, b_elmo_ids, b_labels, b_input_mask = batch
            model.zero_grad()

            loss = model(b_bert_ids, b_elmo_ids, b_labels, b_input_mask.byte())

            loss.backward()

            total_loss += loss.item()

            if show_info and (step+1) % 10 == 0:
                print(f"\n{step}: avg loss per batch: {total_loss/(step*model.num_heads)}\n")

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                            max_norm=max_grad_norm)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader) / model.num_heads
        if show_info:
            print(f"Average train loss: {avg_train_loss}")

        loss_values.append(avg_train_loss)

        ########## VALIDATION ############
        if valid_dataloader is not None:
            head_result, mean_loss, mean_acc, mean_f1 = eval_model(model, valid_dataloader, device, conll)
            
            for head in model.heads.keys():
                head_results[head]['accs'].append(head_result[head]['acc'])
                head_results[head]['f1'].append(head_result[head]['f1'])

            if show_info:
                print(f"Mean validation loss: {mean_loss/model.num_heads}")
                print(f"Mean validation accuracy: {mean_acc/model.num_heads}")
                print(f"Mean validation F1-score: {mean_f1/model.num_heads}\n")
        
        ########## MODEL SAVING ###########
        if save_model and (e+1)%10 == 0:
            #bert_tokenizer.save_pretrained(path_to_save+f'BEbic_{e}_tokenizer_{ver}.pth')
            checkpoint = {'epoch': e,
                          'model': model,
                          'state_dict': model.state_dict(), 
                          'optimizer' : optimizer.state_dict()}
            checkpoint['model_parameters'] = model.get_model_pars_dict()

            torch.save(checkpoint, path_to_save+f'BEbic_{e}_state_dict_{ver}.pth')
    
    if valid_dataloader is not None:
        return loss_values, head_results
    else:
        return loss_values


def load_checkpoint(checkpoint_path, tokenizer_path=None):
    """
    Loads both tokenizer and our pretrained model
    
    Returns:
    -------
    bert_tokenizer
    model
    optimizer state_dict
    model parameters dict
    
    """
    if tokenizer_path is not None:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])

    #optimizer.load_state_dict(checkpoint['optimizer'])
    #for parameter in model.parameters():
    #    parameter.requires_grad = False

    model.eval()
    if tokenizer_path is not None:
        return tokenizer, model, checkpoint['optimizer'], checkpoint['model_parameters']
    else:
        return model, checkpoint['optimizer'], checkpoint['model_parameters']

    
def eval_old(model, dataloader, device, idx2tag):
    model.eval()

    eval_loss = 0
    predictions, true_labels = [], []

    for batch in dataloader:
        if device.type != 'cpu':
            batch = tuple(t.to(device) for t in batch)
        b_bert_ids, b_elmo_ids, b_labels, b_input_mask = batch

        with torch.no_grad():
            logits = model.forward(b_bert_ids, b_elmo_ids, b_input_mask.byte())
            loss = -1*model.crf.forward(logits, b_labels, mask=b_input_mask.byte())
            tags = model.crf.decode(logits, mask=b_input_mask.byte())

        # move loss to cpu
        eval_loss += loss.item()
        predictions.extend(tags)
        labels_ = b_labels.detach().cpu().numpy()
        true_labels.extend(labels_)

    eval_loss = eval_loss / len(dataloader)

    all_predicted_tags = []
    for s in predictions:
        tag_names = [idx2tag[i] for i in s]
        all_predicted_tags.append(tag_names)

    all_true_tags = []
    for s in true_labels:
        tag_names = [idx2tag[i] for i in s if idx2tag[i] != 'PAD']
        all_true_tags.append(tag_names)

    acc = accuracy_score(all_predicted_tags, all_true_tags)
    f1 = f1_score(all_predicted_tags, all_true_tags)
    return eval_loss, acc, f1
    


def train_old(model, train_dataloader, optimizer, device, scheduler=None, n_epoch=5,
              max_grad_norm=None, validate=True, valid_dataloader=None,
              show_info=True, save_model=True):
    loss_values = []
    if validate and valid_dataloader is not None:
        validation_loss_values = []
        valid_accuracies = []
        valid_f1_scores = []

    for e in range(n_epoch):
        if show_info:
          print(f"\nEpoch #{e}")
        # Training

        model.train()

        total_loss = 0

        if show_info:
            enumerator = enumerate(tqdm.tqdm(train_dataloader, position=0, leave=True))
        else:
            enumerator = enumerate(train_dataloader)

        for step, batch in enumerator:
            if device.type != 'cpu':
                batch = tuple(t.to(device) for t in batch)
            b_bert_ids, b_elmo_ids, b_labels, b_input_mask = batch
            model.zero_grad()

            logits = model.forward(b_bert_ids, b_elmo_ids, b_input_mask.byte())
            
            # because we need negative log likelyhood
            loss = -1*model.crf.forward(logits, b_labels, mask=b_input_mask.byte())

            loss.backward()

            total_loss += loss.item()

            if show_info and (step+1) % 10 == 0:
                print(f"\n{step}: avg loss per batch: {total_loss/step}\n")

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                            max_norm=max_grad_norm)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        if show_info:
            print(f"Average train loss: {avg_train_loss}")

        loss_values.append(avg_train_loss)

        if validate and valid_dataloader is not None:
            # Validation
            eval_loss, valid_acc, valid_f1 = eval_old(model, valid_dataloader, device, idx2tag)
            if show_info:
                print(f"Validation loss: {eval_loss}")
                print(f"Validation accuracy: {valid_acc}")
                print(f"Validation F1-score: {valid_f1}\n")
            validation_loss_values.append(eval_loss)
            valid_accuracies.append(valid_acc)
            valid_f1_scores.append(valid_f1)
            
            
        if save_model and (e+1)%10 == 0:
            tokenizer.save_pretrained(f'/content/drive/My Drive/models/ElMo_BERT_biLSTM_oneCRF_{e}_tokenizer.pth')
            checkpoint = {'model': BEboC(hidden_size=512, bert_layers=2),
                          'state_dict': model.state_dict(), 
                          'optimizer' : optimizer.state_dict()}

            torch.save(checkpoint,
                        f'/content/drive/My Drive/models/ElMo_BERT_biLSTM_oneCRF_{e}_state_dict.pth')

    return loss_values, validation_loss_values, valid_accuracies, valid_f1_scores

