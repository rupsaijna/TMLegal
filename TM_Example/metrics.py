def calculate_metrics(outputs, labels):

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

def calculate_precision(metrics):
    return (metrics["tp"])/(metrics["tp"] + metrics["fp"])

def calculate_recall(metrics):
    return (metrics["tp"])/(metrics["tp"] + metrics["fn"])

def calculate_f1(metrics):
    precision = calculate_precision(metrics)
    recall = calculate_recall(metrics)
    return (2 * precision * recall)/(precision + recall)

def get_metrics(metrics):
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

def calculate_document_accuracy(document_outputs, document_labels):

    acc = []
    
    for idx in range(len(document_outputs)):
        correct = True
        for i in range(len(document_outputs[idx])):
            if document_outputs[idx][i] == document_labels[idx][i]:
                continue
            else:
                correct = False
        
        if correct:
            acc.append(1)
        else:
            acc.append(0)
    
    doc_acc = sum(acc)/len(acc)

    return round(doc_acc, 4)

def calculate_chunk_accuracy(chunk_outputs, chunk_labels):

    acc = []

    shape = chunk_outputs.shape
    
    for i in range(len(chunk_outputs)):
        if chunk_outputs[i] == chunk_labels[i]:
            acc.append(1)
        else:
            acc.append(0)
    
    chunk_acc = sum(acc)/len(acc)

    return round(chunk_acc, 4)

def calculate_soft_document_accuracy(document_outputs, document_labels):
    acc = []
    
    for idx in range(len(document_outputs)):
        corr_pred = 0
        ucorr_pred = 0
        
        for i in range(len(document_outputs[idx])):
            if document_outputs[idx][i] == 1 and document_labels[idx][i] == 1:
                corr_pred += 1
            elif document_outputs[idx][i] == 0 and document_labels[idx][i] == 1:
                ucorr_pred += 1
        
        if corr_pred >= ucorr_pred:
            doc_pred = 1
        else:
            doc_pred = 0
        
        if doc_pred == 1 and sum(document_labels[idx]) >= 1:
            acc.append(1)
        elif doc_pred == 0 and sum(document_labels[idx]) >= 1:
            acc.append(0)
            
        
    
    soft_doc_acc = sum(acc)/len(acc)
    
    return round(soft_doc_acc, 4)