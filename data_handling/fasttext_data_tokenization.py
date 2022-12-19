# File includes
from data_handling.data_tokenization import create_data_batch, create_dict_from_json, get_labels_for_document, create_vocabulary
from data_handling.data import DocumentData, BinaryCUADDataset

# Pip includes
from tqdm import tqdm

# Normal includes
import random
import json
import itertools
import os
import math

def create_subparts_fasttext(text: str, subpart_size: int, subpart_overlap: int, vocab_to_idx: dict, tokenized_context: list):
    text_list = tokenized_context

    subpart_size_without_overlap = subpart_size - subpart_overlap

    total_subparts = math.ceil(len(text_list) / (subpart_size - subpart_overlap))
    new_text_list = []
    new_idx_list = []

    for i in range(total_subparts):
        temp_list = [t for t in text_list[i * subpart_size_without_overlap : (i + 1) * subpart_size_without_overlap + subpart_overlap] if not t.is_space]
        if len(temp_list) != subpart_size:
            new_text  = [t.text for t in text_list[i * subpart_size_without_overlap:(i + 1) * subpart_size_without_overlap + subpart_overlap] if not t.is_space] + ["[PAD]" for i in range(subpart_size - len(temp_list))]
            new_text_list.append(new_text)
            
            # Her må jeg legge til at den returnerer kun tekst, FastText skal ikke ha inn integere.

            new_idx = [(int(t.idx), int(t.idx + len(t.text))) for t in text_list[i * subpart_size_without_overlap:(i + 1) * subpart_size_without_overlap + subpart_overlap] if not t.is_space] + [(0, 0) for i in range(subpart_size - len(temp_list))]
            new_idx_list.append(new_idx)
        else:
            new_text = [t.text for t in text_list[i * subpart_size_without_overlap:(i + 1) * subpart_size_without_overlap + subpart_overlap] if not t.is_space]
            new_text_list.append(new_text)

            new_idx = [(t.idx, t.idx + len(t.text)) for t in text_list[i * subpart_size_without_overlap:(i + 1) * subpart_size_without_overlap + subpart_overlap] if not t.is_space]
            new_idx_list.append(new_idx)

        
    return new_text_list, new_idx_list

def create_dataset_fasttext(datasource: str, datadestination: str, vocab_destination: str, num_datapoints: int, subpart_size: int, subpart_overlap: int, tokenize: any) -> dict:
    data = create_dict_from_json(datasource)

    # Must check if the data has been created already

    vocab_to_idx, tokenized_contexts, vocab_size = create_vocabulary(tokenize, data, vocab_destination, num_datapoints)

    fpath = datadestination

    if os.path.exists(fpath):
        print("Found existing file, loading....")
        data = json.load(open(datadestination))
        print("Finished loading file")
    else:  
        print("No existing file with same configuration, creating new file....")
        # Create subparts and labels for each category
        for _, filename in tqdm(enumerate(dict(itertools.islice(data.items(), num_datapoints))), total=len(dict(itertools.islice(data.items(), num_datapoints))), desc="Creating subparts and labels"):
            data[filename]["subparts_tokens"], data[filename]["subparts_idx"] = create_subparts_fasttext(data[filename]["context"], subpart_size, subpart_overlap, vocab_to_idx, tokenized_contexts[filename])
            data[filename]["labels"] = get_labels_for_document(data[filename], data[filename]["subparts_idx"])

        with open(fpath, 'w') as fp:
            json.dump(data, fp)
    
    return data, vocab_to_idx, vocab_size


def get_dataset_for_category_fasttext(category: str, data_source, data_destination, vocab_destination, num_examples, subpart_size, subpart_overlap, tokenize):

    data, vocab_to_idx, vocab_size = create_dataset_fasttext(data_source, data_destination, vocab_destination, num_examples, subpart_size, subpart_overlap, tokenize)

    documents = []
    for filename in itertools.islice(data, num_examples):
        documents.append(DocumentData(data[filename]["subparts_tokens"], data[filename]["labels"][category], filename))
    
    dataset = BinaryCUADDataset(documents)
    
    positive_datapoints = []
    negative_datapoints = []

    for d in dataset:
        if sum(d["labels"]) == 0:
            negative_datapoints.append(d)
        else:
            positive_datapoints.append(d)

    split_ratio = 0.7

    train_data = positive_datapoints[:int(len(positive_datapoints) * split_ratio)]
    train_data.extend(negative_datapoints[:int(len(negative_datapoints) * split_ratio)])
    
    train_data_pos = positive_datapoints[:int(len(positive_datapoints) * split_ratio)]
    train_data_neg = negative_datapoints[:int(len(negative_datapoints) * split_ratio)]

    test_data = positive_datapoints[int(len(positive_datapoints) * split_ratio):]
    test_data.extend(negative_datapoints[int(len(negative_datapoints) * split_ratio):])
    
    tmp_train_pos = []
    tmp_train_neg = []
    
    total_nr_of_chunks = 0
    
    for doc in train_data:
        for chunk_id in range(len(doc["labels"])):
            total_nr_of_chunks += 1
            if doc["labels"][chunk_id] == 1:
                tmp_train_pos.append({"subpart": doc["subparts"][chunk_id], "label": doc["labels"][chunk_id]})
            else:
                tmp_train_neg.append({"subpart": doc["subparts"][chunk_id], "label": doc["labels"][chunk_id]})
    
    pos_doc_count = 0
    neg_doc_count = 0
    
    for d in train_data:
        if sum(d["labels"]) > 0:
            pos_doc_count += 1
        else:
            neg_doc_count += 1
    
    cat_pos_neg_ratio = round(pos_doc_count / (neg_doc_count + pos_doc_count), 1)
    
    if cat_pos_neg_ratio > 0.7:
        cat_pos_neg_ratio = 0.7
    elif cat_pos_neg_ratio < 0.3:
        cat_post_neg_ratio = 0.3
    
    print("Ratio:", cat_pos_neg_ratio)

    print("Pos train:", len(positive_datapoints))
    print("Neg train:", len(negative_datapoints))
    
    batch_size = total_nr_of_chunks//num_examples
    
    print("Batch size:", batch_size)
    
    train_data = []
    
    # For positive examples
    for i in range(num_examples):
        batch = create_data_batch(batch_size, tmp_train_pos, tmp_train_neg, cat_pos_neg_ratio)
        random.shuffle(batch)
        train_data.append(batch)
        
    print("Randomizing training and test data..")
    
    random.shuffle(train_data)
    random.shuffle(test_data)
        
    if not os.path.exists(f"./data/spacy_tokenize_{num_examples}_{subpart_size}_{subpart_overlap}"):
        tokenize.to_disk(f"./data/spacy_tokenize_{num_examples}_{subpart_size}_{subpart_overlap}")
    else:
        tokenize = tokenize.from_disk(f"./data/spacy_tokenize_{num_examples}_{subpart_size}_{subpart_overlap}")
    
    return train_data, test_data, tokenize, vocab_to_idx