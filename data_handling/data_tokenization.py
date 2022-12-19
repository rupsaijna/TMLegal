import math
import json
from typing import Tuple
import itertools
import os
import random

from data_handling.data import DocumentData, BinaryCUADDataset

import spacy
import unidecode
from tqdm import tqdm


def create_vocabulary(tokenize: any, data: dict, vocab_destination: str, num_datapoints: int) -> list:

    combined_contexts = ["[PAD]"]
    

    token_to_idx = {}

    tokenized_contexts = {}

    # Store the tokenized contexts and pass them along

    if os.path.exists(vocab_destination):
        print("Found existing file, loading....")
        token_to_idx = json.load(open(vocab_destination))
        print("Finished loading file")
        #for _, filename in tqdm(enumerate(dict(itertools.islice(data.items(), num_datapoints))), total=len(dict(itertools.islice(data.items(), num_datapoints))), desc="Only tokenization of contexts"):
        #    tokenized_context = tokenize(data[filename]["context"])
        #    tokenized_contexts[filename] = tokenized_context
    else:
        for _, filename in tqdm(enumerate(dict(itertools.islice(data.items(), num_datapoints))), total=len(dict(itertools.islice(data.items(), num_datapoints))), desc="Creating vocabulary"):
            tokenized_context = tokenize(data[filename]["context"])
            tokenized_contexts[filename] = tokenized_context
            for token in tokenized_context:
                if not token.is_space:
                    combined_contexts.append(token.text)
    

        for idx, token in tqdm(enumerate(list(set(combined_contexts))), total=len(combined_contexts), desc="Creating Token to Index Mapping"):
            if token in token_to_idx:
                continue
            else:
                token_to_idx[token] = idx

        with open(vocab_destination, 'w') as fp:
            json.dump(token_to_idx, fp)
    
    vocab_size = len(token_to_idx)

    return token_to_idx, tokenized_contexts, token_to_idx

def create_subparts(text: str, subpart_size: int, subpart_overlap: int, vocab_to_idx: dict, tokenized_context: list) -> Tuple[list, list]:
    text_list = tokenized_context

    subpart_size_without_overlap = subpart_size - subpart_overlap

    total_subparts = math.ceil(len(text_list) / (subpart_size - subpart_overlap))
    new_text_list = []
    new_idx_list = []

    for i in range(total_subparts):
        temp_list = [t for t in text_list[i * subpart_size_without_overlap : (i + 1) * subpart_size_without_overlap + subpart_overlap] if not t.is_space]
        if len(temp_list) != subpart_size:
            new_text  = [vocab_to_idx[t.text] for t in text_list[i * subpart_size_without_overlap:(i + 1) * subpart_size_without_overlap + subpart_overlap] if not t.is_space] + [vocab_to_idx["[PAD]"] for i in range(subpart_size - len(temp_list))]
            new_text_list.append(new_text)
            
            # Her må jeg legge til at den returnerer kun tekst, FastText skal ikke ha inn integere.

            new_idx = [(int(t.idx), int(t.idx + len(t.text))) for t in text_list[i * subpart_size_without_overlap:(i + 1) * subpart_size_without_overlap + subpart_overlap] if not t.is_space] + [(0, 0) for i in range(subpart_size - len(temp_list))]
            new_idx_list.append(new_idx)
        else:
            new_text = [vocab_to_idx[t.text] for t in text_list[i * subpart_size_without_overlap:(i + 1) * subpart_size_without_overlap + subpart_overlap] if not t.is_space]
            new_text_list.append(new_text)

            new_idx = [(t.idx, t.idx + len(t.text)) for t in text_list[i * subpart_size_without_overlap:(i + 1) * subpart_size_without_overlap + subpart_overlap] if not t.is_space]
            new_idx_list.append(new_idx)

        
    return new_text_list, new_idx_list

def get_all_categories_with_answers(ex_dict):

    categories = []

    for category in ex_dict["categories"]:
        if len(ex_dict["categories"][category]) > 0:
            categories.append(category)
    
    return categories


def get_labels_for_document(ex_dict: dict, subparts_idx: list) -> dict:
    labels = {}

    for category, answers in ex_dict["categories"].items():
        labels[category] = []
        categories_with_labels = get_all_categories_with_answers(ex_dict)
        if category in categories_with_labels:
            for subpart_idx in subparts_idx:
                occurring = False
                for start, end in subpart_idx:
                    for ans in answers:
                        if start == ans["start"] or end == ans["end"]:
                            occurring = True
                if occurring:
                    labels[category].append(1)
                else:
                    labels[category].append(0)
        else:
            labels[category] = [0 for i in range(len(subparts_idx))]
    
    return labels

def normalize_title(title: str) -> str:
    title = title.lower()
    title = title.replace(" ", "")
    title = title.replace("-", "")
    title = title.replace("_", "")
    title = title.replace("'", "")
    title = title.replace(".", "")
    title = unidecode.unidecode(title)
    return title

def create_dict_from_json(json_path: str) -> dict:
    data = json.load(open(json_path))["data"]

    my_data = {}

    for i, document in enumerate(data):
        document_title = document["title"]
        my_data[document_title] = {}
        my_data[document_title]["context"] = document["paragraphs"][0]["context"]
        my_data[document_title]["categories"] = {}
        for qas in document["paragraphs"][0]["qas"]:
            category_name = qas["id"].split("__")[1].lower().replace(" ", "_").replace("/", "")
            my_data[document_title]["categories"][category_name] = []
            if not qas["is_impossible"]:
                for answer in qas["answers"]:
                    text = answer["text"]
                    text_start = int(answer["answer_start"])
                    text_end = int(text_start + len(text))
                    my_data[document_title]["categories"][category_name].append({"text": text, "start": text_start, "end": text_end})
    
    return my_data
            

def create_dataset(datasource: str, datadestination: str, vocab_destination: str, num_datapoints: int, subpart_size: int, subpart_overlap: int, tokenize: any) -> dict:
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
            data[filename]["subparts_tokens"], data[filename]["subparts_idx"] = create_subparts(data[filename]["context"], subpart_size, subpart_overlap, vocab_to_idx, tokenized_contexts[filename])
            data[filename]["labels"] = get_labels_for_document(data[filename], data[filename]["subparts_idx"])

        with open(fpath, 'w') as fp:
            json.dump(data, fp)
    
    return data, vocab_to_idx, vocab_size


def get_dataset_for_category(category: str, data_source, data_destination, vocab_destination, num_examples, subpart_size, subpart_overlap, tokenize):

    data, vocab_to_idx, vocab_size = create_dataset(data_source, data_destination, vocab_destination, num_examples, subpart_size, subpart_overlap, tokenize)

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

def create_data_batch(batch_size, positive_examples, negative_examples, pos_neg_ratio):
    tmp = []
    for i in range(batch_size):
        if i <= int(batch_size * pos_neg_ratio):
            tmp.append(random.choice(positive_examples))
        else:
            tmp.append(random.choice(negative_examples))
    
    return tmp

if __name__ == '__main__':
    pass