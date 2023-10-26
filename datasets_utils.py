import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch
import json
import datasets
from datasets import Dataset as Dataset2


def load_data_text(
    batch_size, 
    seq_len, 
    deterministic=False, 
    data_args=None, 
    model_emb=None,
    split='train', 
    loaded_vocab=None,
    loop=True,
):
    """
    For a dataset, create a generator over (seqs, kwargs) pairs.

    Each seq is an (bsz, len, h) float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for some meta information.

    -> batch_size: the batch size of each returned pair.
    -> seq_len: the max sequence length (one-side).
    -> deterministic: if True, yield results in a deterministic order.
    -> data_args: including dataset directory, num of dataset, basic settings, etc.
    -> model_emb: loaded word embeddings.
    -> loaded_vocab: loaded word vocabs.
    -> loop: loop to get batch data or not.
    """
    print("Loading text data", "."*10)

    training_data = get_corpus(data_args, seq_len, split=split, loaded_vocab=loaded_vocab)


def get_corpus(data_args, seq_len, split='train', loaded_vocab=None):
    """
    Obtain the given split of data from the data directory.
    """
    print(f"Loading dataset {data_args.dataset} from {data_args.data_dir} ..........")
    
    sent2sent_list = {'src':[], 'trg':[]}

    if split == 'train':
        print("Loading data from the training set.....")
        path = f'{data_args.data_dir}/train.jsonl'
    elif split == 'val':
        print("Loading data from the val set.....")
        path = f'{data_args.data_dir}/val.jsonl'
    elif split == 'test':
        print("Loading data from the test set.....")
        path = f'{data_args.data_dir}/test.jsonl'
    else:
        assert False, "invalid split for dataset"
    
    with open(path, 'r') as f_reader:
        for row in f_reader:
            content = json.loads(row)
            sent2sent_list['src'].append(content['src'].strip())
            sent2sent_list['trg'].append(content['trg'].strip())
    
    vocab_dict = loaded_vocab
    train_data = helper_tokenize(sent2sent_list, vocab_dict, seq_len)
    return train_data


def helper_tokenize(sent_list, vocab_dict, seq_len):
    """
    """
    raw_dataset = Dataset2.from_dict(sent_list)

    def tokenize_function(examples):
        """
        Used in the map() function below.
        Takes data from the dataset in batches and tokenizes them.
        """
        input_id_x = vocab_dict.encode_token(examples['src'])
        input_id_y = vocab_dict.encode_token(examples['trg'])
        result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}
        return result_dict

    tokenized_datasets = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    print('### tokenized_datasets', tokenized_datasets)
    print('### tokenized_datasets...example', tokenized_datasets['input_id_x'][0])

    def merge_and_mask(group_lst):
        """
        When we pass input, we concatenate src and trg (as mentioned in the diffuseq paper).
        The combined input needs to be less than or equal to the max_seq_len allowed 
        by the underlying transformer model.
        """
        lst = []
        mask = []
        for i in range(len(group_lst['input_id_x'])):
            end_token = group_lst['input_id_x'][i][-1]
            src = group_lst['input_id_x'][i][:-1]
            trg = group_lst['input_id_y'][i][:-1]
            """
            go on popping from the end of the source and target sentences one by one till
            combined length of source and target = seq_len-3 is reached
            """
            while len(src) + len(trg) > seq_len - 3:
                if len(src)>len(trg):
                    src.pop()
                elif len(src)<len(trg):
                    trg.pop()
                else:
                    src.pop()
                    trg.pop()
            src.append(end_token)
            trg.append(end_token)

            lst.append(src + [vocab_dict.sep_token_id] + trg)
            mask.append([0]*(len(src)+1))
        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask
        return group_lst