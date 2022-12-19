# File includes
from metrics.metrics import get_metrics, calculate_metrics, calculate_chunk_accuracy, calculate_document_accuracy, calculate_soft_document_accuracy
from data_handling.data_tokenization import get_dataset_for_category

# Pip includes
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
#import bcolz
import pickle
import spacy
import torch.nn as nn
import numpy as np
import pandas as pd


# Normal includes
import json
import os

def store_glove_vectors(glove_path):
    words = []
    idx = 0
    word2idx = {}
    vectors = np.zeros(1)
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

def get_train_test_dataset_for_category(category, num_examples=255, subpart_size=256, subpart_overlap=30):
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



def create_embedding_matrix(vocab, embedding_dict, dimension):
    embedding_matrix=np.zeros((len(vocab) + 1, dimension))
 
    for word, index in vocab.items():
        if word in embedding_dict:
            embedding_matrix[index]=embedding_dict[word]
    return embedding_matrix

class LSTM(nn.Module):

    def __init__(self, dimension, vocab, lstm_layers, seq_len, cnn_hidden):
        super(LSTM, self).__init__()
        
        
        # FAST glove embedding because bcolz let me down after server upgrade
        print("Reading in glove vector")
        glove = pd.read_csv('glove.6B.300d.txt', sep=" ", quoting=3, header=None, index_col=0)
        glove_embedding = {key: val.values for key, val in glove.T.items()}
        print("Finished reading glove vector")
        
        self.embedding_dim = 300
        
        embedding_matrix = create_embedding_matrix(vocab, glove_embedding, dimension=self.embedding_dim)
        
        vocab_size=embedding_matrix.shape[0]
        vector_size=embedding_matrix.shape[1]
 
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=vector_size)
        
        self.embedding.weight=nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float32))


        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=dimension,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True)
        
        self.drop = nn.Dropout(p=0.5)
        
        self.conv_drop = nn.Dropout(p=0.15)

        # CNN PARAMS
        self.conv_filter_1 = 3
        self.conv_filter_2 = 5
        self.stride = 2
        
        # CONVOLUTION LAYERS
        self.conv_1 = nn.Conv1d(in_channels=1, out_channels=cnn_hidden, kernel_size=self.conv_filter_1, stride=self.stride, padding=4)
        self.conv_2 = nn.Conv1d(in_channels=1, out_channels=cnn_hidden, kernel_size=self.conv_filter_2, stride=self.stride, padding=6)

        # POOLING LAYERS
        self.pool_1 = nn.MaxPool1d(kernel_size=self.conv_filter_1, stride=6, padding=self.conv_filter_1//2)
        self.pool_2 = nn.MaxPool1d(kernel_size=self.conv_filter_2, stride=10, padding=self.conv_filter_2//2)

        # SEQUENCE LEN
        self.seq_len = seq_len

        self.relu = nn.ReLU()

        # PREDICTION LAYER
        self.fc_1 = nn.Linear(dimension * cnn_hidden * 2, dimension)
        self.fc_2 = nn.Linear(dimension, 1)

    def forward(self, text):

        text_emb = self.embedding(text)

        text_emb = text_emb.unsqueeze(0)

        output, _ = self.lstm(text_emb)

        out_forward = output[range(len(output)), self.seq_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        #print("Text fea:", text_fea.shape)

        x1, x2 = self.conv(text_fea)

        #print("x1, x2:", x1.shape, x2.shape)
        

        output = torch.cat((x1, x2), dim=0)

        #print("output:", output.shape)
        output = output.view(output.shape[2], output.shape[0] * output.shape[1])
        #print("output:", output.shape)

        output = self.fc_1(output)
        output = self.relu(output)
        output = self.drop(output)
        output = self.fc_2(output)
        output = torch.squeeze(output, 1)
        output = torch.sigmoid(output)

        return output
    
    def conv(self, inp):

        inp = torch.reshape(inp, (self.dimension, 1, 2))

        x1 = self.conv_1(inp)
        x2 = self.conv_2(inp)

        x1 = self.relu(x1)
        x2 = self.relu(x2)
        
        x1 = self.conv_drop(x1)
        x2 = self.conv_drop(x2)

        #print("x1:", x1.shape)
        #print("x2:", x2.shape)

        x1 = self.pool_1(x1)
        x2 = self.pool_2(x2)

        return x1, x2


# Training Function

def train(model,
          optimizer,
          train_loader,
          test_loader,
          category,
          criterion = nn.BCELoss(),
          num_epochs = 5,
          file_path = "./models",
          best_valid_loss = float("Inf"),
          device = "cpu"):
    
    results = {"train_loss": [], "test_loss": [], "soft_doc_accs": [], "metrics": [], "chunk_acc": [], "doc_acc": []}
    
    # initialize running values
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    soft_doc_accs = []
    global_steps_list = []

    # training loop
    model.train()
    for _, epoch in tqdm(enumerate(range(num_epochs)), total=num_epochs, desc="Training Epochs"):
        running_loss = 0.0
        train_outputs = []
        train_labels = []
        for _, sample in enumerate(train_loader):
            
            labels = []
            inputs = []
            for s in sample:
                labels.append(torch.FloatTensor([float(s["label"])]).to(device))
                inputs.append(torch.LongTensor(s["subpart"]).to(device))

            outputs = []
            
            for inp in inputs:
                output = model(inp)
                outputs.append(output)

            train_outputs.extend(outputs)
            train_labels.extend(labels)
            
            labels = torch.cat(labels, dim=0)

            if len(outputs) > 1:
                outputs = torch.cat(outputs, dim=0).squeeze()
            else:
                outputs = torch.cat(outputs, dim=0)
            
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            #print(loss.item())

            # evaluation step

        print()
        print()
        train_outputs = torch.cat(train_outputs)
        print(f"Training ({epoch}/{num_epochs}):", get_metrics(calculate_metrics(train_outputs, train_labels)), "Loss:", round(running_loss/len(train_loader), 5))
<<<<<<< HEAD
        valid_loss, soft_doc_acc, metrics, chunk_acc, doc_acc = evaluation(model, test_loader, criterion, device)
=======
        valid_loss, soft_doc_acc, metrics, chunk_acc, doc_acc = evaluation(model, test_loader, criterion, device, category, finaleval)
>>>>>>> updating results
        model.train()
        
        train_loss_list.append(running_loss/len(train_loader))
        valid_loss_list.append(valid_loss)
        soft_doc_accs.append(soft_doc_acc)
        
        results["train_loss"].append(running_loss/len(train_loader))
        results["test_loss"].append(valid_loss)
        results["soft_doc_accs"].append(soft_doc_acc)
        results["metrics"].append(metrics)
        results["chunk_acc"].append(chunk_acc)
        results["doc_acc"].append(doc_acc)
        
        #print("Epoch", epoch, ":", round(running_loss/len(train_loader), 5))
        
    
    #save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')
    plt.plot(train_loss_list, label="Train")
    plt.plot(valid_loss_list, label="Test")
    plt.title("Train Loss vs Test Loss")
    plt.legend()
    plt.savefig(f"./results/bilstmcnn/{category}_{num_epochs}_train_test_loss.png")
    plt.show()
    
    plt.plot(soft_doc_accs)
    plt.title("Soft Document Accuracy")
    plt.savefig(f"./results/bilstmcnn/{category}_{num_epochs}_soft_doc_acc.png")
    plt.show()
<<<<<<< HEAD
=======

    finaleval=1
    valid_loss, soft_doc_acc, metrics, chunk_acc, doc_acc = evaluation(model, test_loader, criterion, device, category, finaleval)
>>>>>>> updating results
    
    with open(f"./results/bilstmcnn/{category}_{num_epochs}.json", 'w') as fp:
            json.dump(results, fp)
            
    torch.save(model, f"./models/bilstmcnn/bilstm_cnn_{category}_{num_epochs}.pt")
    

<<<<<<< HEAD
def evaluation(model, test_loader, criterion, device):
=======
def evaluation(model, test_loader, criterion, device, category, finaleval):
>>>>>>> updating results
    model.eval()
    valid_running_loss = 0.0
    with torch.no_grad():                    
        # validation loop
        document_eval_outputs = []
        document_eval_labels = []
        eval_outputs = []
        eval_labels = []
<<<<<<< HEAD
=======
        eval_probs = []
        normal_evals_labels=[]
>>>>>>> updating results
        for _, eval_sample in enumerate(test_loader):
            labels = torch.FloatTensor(eval_sample["labels"]).to(device)
            inputs = torch.LongTensor(eval_sample["subparts"]).to(device)

            outputs = []

            for inp in inputs:
                output = model(inp)
                outputs.append(output)

            eval_outputs.extend(outputs)
            eval_labels.extend(labels)
<<<<<<< HEAD
=======
            eval_probs.extend(probs)
            #normal_evals_labels+=[eval_sample["labels"]]
            normal_evals_labels.extend(labels)
>>>>>>> updating results

            outputs = torch.cat(outputs)
            
            loss = criterion(outputs, labels)
            
            valid_running_loss += loss.item()

            document_eval_outputs.append(outputs)
            document_eval_labels.append(labels)
        eval_outputs = torch.cat(eval_outputs)
<<<<<<< HEAD
=======
        eval_probs = torch.cat(eval_probs)
        normal_evals_labels = torch.stack(normal_evals_labels)
>>>>>>> updating results

        print()
        print("---------- EVALUATION ----------")
        print("Document Accuracy:", calculate_document_accuracy(document_eval_outputs, document_eval_labels))
        print("Soft Document Accuracy:", calculate_soft_document_accuracy(document_eval_outputs, document_eval_labels))
        print("Chunk Accuracy:", calculate_chunk_accuracy(eval_outputs, eval_labels))

        print("Loss:", valid_running_loss/len(test_loader))
        metrics = get_metrics(calculate_metrics(eval_outputs, eval_labels))
        for k, v in metrics.items():
            print(f"{k}:", v)
<<<<<<< HEAD
=======

        if finaleval==1:
            prfs = precision_recall_fscore_support(np.array(normal_evals_labels.cpu()).tolist(), np.array(torch.round(eval_outputs).cpu(), dtype=int).tolist(), average='weighted')
            with open(f"results2/bilstmcnn/{category}_metrics.txt", 'w') as fp:
                fp.write(str(prfs))
            with open(f"results2/bilstmcnn/{category}_probabilityscores.txt", 'w') as fp:
                fp.write(str(np.array(eval_probs.cpu()).tolist()))
            with open(f"results2/bilstmcnn/{category}_truthvalues.txt", 'w') as fp:
                fp.write(str(np.array(normal_evals_labels.cpu()).tolist()))
>>>>>>> updating results
        
        return valid_running_loss/len(test_loader), calculate_soft_document_accuracy(document_eval_outputs, document_eval_labels), metrics, calculate_chunk_accuracy(eval_outputs, eval_labels), calculate_document_accuracy(document_eval_outputs, document_eval_labels)