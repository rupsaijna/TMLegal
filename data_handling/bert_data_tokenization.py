# File includes
from data_handling.data import DocumentData, BinaryCUADDataset
from data_handling.data_tokenization import create_dict_from_json, create_data_batch, get_labels_for_document

# Pip includes
from tqdm import tqdm

# Normal includes
import os
import json
import itertools
import random
import math


def create_vocabulary_bert(tokenize, data, num_datapoints):

    tokenized_contexts = {}

    for _, filename in tqdm(enumerate(dict(itertools.islice(data.items(), num_datapoints))), total=len(dict(itertools.islice(data.items(), num_datapoints))), desc="Creating vocabulary"):
        tokenized_context = tokenize(data[filename]["context"], return_offsets_mapping=True)
        tokenized_contexts[filename] = tokenized_context
    
    print(len(tokenized_contexts))

    return tokenized_contexts

def create_subparts_bert(text, subpart_size, subpart_overlap):

    char_offsets = text["offset_mapping"]
    
    text = text["input_ids"]

    subpart_size_without_overlap = subpart_size - subpart_overlap

    total_subparts = math.ceil(len(text) / (subpart_size - subpart_overlap))

    new_text_list = []
    new_idx_list = []

    for i in range(total_subparts):
        temp_list = [t for t in text[i * subpart_size_without_overlap : (i + 1) * subpart_size_without_overlap + subpart_overlap]]
        if len(temp_list) != subpart_size:
            new_text  = [t for t in text[i * subpart_size_without_overlap:(i + 1) * subpart_size_without_overlap + subpart_overlap]] + [0 for i in range(subpart_size - len(temp_list))]
            new_text_list.append(new_text)

            new_idx = [t for t in char_offsets[i * subpart_size_without_overlap:(i + 1) * subpart_size_without_overlap + subpart_overlap]] + [(0, 0) for i in range(subpart_size - len(temp_list))]
            new_idx_list.append(new_idx)
        else:
            new_text = [t for t in text[i * subpart_size_without_overlap:(i + 1) * subpart_size_without_overlap + subpart_overlap]]
            new_text_list.append(new_text)

            new_idx = [t for t in char_offsets[i * subpart_size_without_overlap:(i + 1) * subpart_size_without_overlap + subpart_overlap]]
            new_idx_list.append(new_idx)
    
    return new_text_list, new_idx_list

def create_dataset_bert(datasource, datadestination, vocab_destination, num_datapoints, subpart_size, subpart_overlap, tokenize):
    data = create_dict_from_json(datasource)

    tokenized_contexts = create_vocabulary_bert(tokenize, data, num_datapoints)

    fpath = datadestination
        
    if os.path.exists(datadestination):
        print("Found existing file, loading....")
        data = json.load(open(datadestination))
        print("Finished loading file")
    else:
        print("No existing file with same configuration, creating new file....")
        # Create subparts and labels for each category
        for _, filename in tqdm(enumerate(dict(itertools.islice(data.items(), num_datapoints))), total=len(dict(itertools.islice(data.items(), num_datapoints))), desc="Creating subparts and labels"):
            data[filename]["subparts_tokens"], data[filename]["subparts_idx"] = create_subparts_bert(tokenized_contexts[filename], subpart_size, subpart_overlap)
            data[filename]["labels"] = get_labels_for_document(data[filename], data[filename]["subparts_idx"])
        
        print(data)
        with open(fpath, 'w') as fp:
            json.dump(data, fp)
    
    return data


def get_dataset_for_category_bert(category, data_source, data_destination, vocab_destination, num_examples, subpart_size, subpart_overlap, tokenize):
    
    data = create_dataset_bert(data_source, data_destination, vocab_destination, num_examples, subpart_size, subpart_overlap, tokenize)
    
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
    
    print("Randomizing training and testing data..")
    
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    return train_data, test_data, tokenize