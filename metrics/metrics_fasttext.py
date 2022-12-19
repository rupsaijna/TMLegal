from metrics.metrics import calculate_precision, calculate_f1, calculate_recall

def calculate_metrics_ft(outputs, labels):

    tmp_outputs = []
    for subout in outputs:
        for out in subout:
            tmp_outputs.append(int(out))
        
    tmp_labels = []
    for sublab in labels:
        for lab in sublab:
            tmp_labels.append(int(lab))
        
    outputs = tmp_outputs
    labels = tmp_labels
    
    metrics = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for i in range(len(outputs)):
        if outputs[i] == 1 and labels[i] == 1:
            metrics["tp"] += 1
        elif outputs[i] == 0 and labels[i] == 0:
            metrics["tn"] += 1
        elif outputs[i] == 1 and labels[i] == 0:
            metrics["fp"] += 1
        elif outputs[i] == 0 and labels[i] == 1:
            metrics["fn"] += 1
        
    return metrics

def get_metrics_ft(metrics):
    try:
        precision = calculate_precision(metrics)
    except Exception as e:
        precision = None
        print("Couldnt calculate precision, division by zero")
    
    try:
        recall = calculate_recall(metrics)
    except Exception as e:
        recall = None
        print("Couldnt calculate recall, division by zero")
    
    try:
        f1 = calculate_f1(metrics)
    except Exception as e:
        f1 = None
        print("Couldnt calculate f1, division by zero")


    return {"precision": precision, "recall": recall, "f1": f1, "tp": metrics["tp"], "tn": metrics["tn"], "fp": metrics["fp"], "fn": metrics["fn"]}

def calculate_soft_document_accuracy_ft(document_outputs, document_labels):
    acc = []
    
    for idx in range(len(document_outputs)):
        corr_pred = 0
        ucorr_pred = 0
        
        for i in range(len(document_outputs[idx])):
            document_outputs[idx] = [int(o) for o in document_outputs[idx]]
            if document_outputs[idx][i] == 1 and document_labels[idx][i] == 1:
                corr_pred += 1
            elif document_outputs[idx][i] == 0 and document_labels[idx][i] == 1:
                ucorr_pred += 1
        
        if corr_pred >= ucorr_pred:
            pred = 1
        else:
            pred = 0
        
        if pred == 1 and sum(document_labels[idx]) >= 1:
            acc.append(1)
        elif pred == 0 and sum(document_labels[idx]) >= 1:
            acc.append(0)
    
    soft_doc_acc = sum(acc)/len(acc)
    
    return round(soft_doc_acc, 4)

def calculate_chunk_and_doc_accuracy(preds, labels):

    chunk_acc = []
    doc_acc = []

    for i in range(len(preds)):
        if [int(k) for k in preds[i]] == labels[i]:
            doc_acc.append(1)
        else:
            doc_acc.append(0)
        for j in range(len(preds[i])):
            if int(preds[i][j]) == labels[i][j]:
                chunk_acc.append(1)
            else:
                chunk_acc.append(0)

    return round(sum(chunk_acc)/len(chunk_acc), 4), round(sum(doc_acc)/len(doc_acc), 4)

def get_preds_and_labels(test_dataset, model):
    preds = []
    labels = []

    for document in test_dataset:
        for key, item in document.items():
            if key == "labels":
                labels.append(item)
            else:
                tmp = []
                for chunk in item:
                    pred = model.predict(" ".join(chunk))
                    tmp.append(pred)

                preds.append(tmp)

    new_preds = []

    for i in preds:
        doc_preds = []
        for pred in i:
            doc_preds.append(pred[0][0].split("__label__")[1])
        new_preds.append(doc_preds)

    return new_preds, labels