#import bcolz
import pickle

import spacy

import os

from data_handling.data_tokenization import get_dataset_for_category

import torch.nn as nn
import numpy as np
import torch

def store_glove_vectors(glove_path):
    words = []
    idx = 0
    word2idx = {}
    #vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.300.dat', mode='w')

    with open(f'{glove_path}/glove.6B.300d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    #vectors = bcolz.carray(vectors[1:].reshape((400001, 300)), rootdir=f'{glove_path}/6B.300.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{glove_path}/6B.300_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/6B.300_idx.pkl', 'wb'))
    

def create_dict_from_glove(glove_path):
    #vectors = bcolz.open(f'{glove_path}/6B.300.dat')[:]
    words = pickle.load(open(f'{glove_path}/6B.300_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.300_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}
    
    return glove
    

def create_matrix_based_on_vocab(vocab, glove):
    emb_dim = 300
    
    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, emb_dim))
    words_found = 0

    for i, word in enumerate(vocab):
        try: 
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
    
    return weights_matrix
    
def create_emb_layer(vocab, non_trainable=False):
    glove_path = "./glove/"
    
    if not os.path.exists(glove_path + "6B.300.dat"):
        store_glove_vectors(glove_path)
    else:
        print("Glove already created and stored")
        
    glove = create_dict_from_glove(glove_path)
    
    weights_matrix = create_matrix_based_on_vocab(vocab, glove)
    
    weights_matrix = torch.FloatTensor(weights_matrix)
    
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding.from_pretrained(weights_matrix)
    if non_trainable:
        emb_layer.weight.requires_grad = False
    
    return emb_layer, num_embeddings, embedding_dim

def get_train_test_dataset_for_category(category, num_examples, subpart_size, subpart_overlap):
    data_source = "data/CUADv1.json"
    num_examples = num_examples # 510 is max
    subpart_size = subpart_size
    subpart_overlap = subpart_overlap
    data_destination = f"data/binary_dataset_{subpart_size}_{subpart_overlap}_{num_examples}.json"
    vocab_destination = f"data/vocab_{num_examples}.json"
    category = category

    # Save everything the Spacy tokenizer gives us
    tokenize = spacy.load("en_core_web_sm")

    train_dataset, test_dataset, tokenizer, vocab_size = get_dataset_for_category(category, data_source, data_destination, vocab_destination, num_examples, subpart_size, subpart_overlap, tokenize)
    
    return train_dataset, test_dataset, tokenizer, vocab_size

def get_device():
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device

#if __name__ == "__main__":
#    glove_path = "./glove/"
#        
#    print("Storing glove vectors")
#    store_glove_vectors(glove_path)
#    
#    print("Creating dict from glove")
#    glove = create_dict_from_glove(glove_path)
#    
#    string = "Hi my name is Sander. I am very cool."
#    
#    vocab = {idx: word.strip(".").lower() for idx, word in enumerate(string.split(" "))}
#    
#    print("Creating glove matrix")
#    glove_matrix = create_matrix_based_on_vocab(vocab, glove)