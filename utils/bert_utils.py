# File includes
from metrics.metrics import calculate_metrics, get_metrics, calculate_chunk_accuracy, calculate_document_accuracy, calculate_soft_document_accuracy

# Pip includes
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Normal includes
import json

def train_bert(model,
          optimizer,
          train_loader,
          test_loader,
          category,
          model_name,
          criterion = nn.BCELoss(),
          num_epochs = 5,
          file_path = "./models",
          best_valid_loss = float("Inf"),
          device = "cuda"):
    
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
        for s_idx, sample in enumerate(train_loader):
            
            labels = []
            inputs = []
            for s in sample:
                labels.append(torch.FloatTensor([float(s["label"])]).to(device))
                inputs.append(torch.LongTensor(s["subpart"]).to(device))

            outputs = []
            
            for idx in range(len(inputs)):
                output = torch.sigmoid(model(inputs[idx].unsqueeze(0)).logits)
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

        print()
        print()
        train_outputs = torch.cat(train_outputs)
        print(f"Training ({epoch}/{num_epochs}):", get_metrics(calculate_metrics(train_outputs, train_labels)), "Loss:", round(running_loss/len(train_loader), 5))
        valid_loss, soft_doc_acc, metrics, chunk_acc, doc_acc = evaluation_bert(model, test_loader, criterion, device)
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
    plt.savefig(f"./results/{model_name}/{category}_{num_epochs}_train_test_loss.png")
    plt.show()
    
    plt.plot(soft_doc_accs)
    plt.title("Soft Doc Acc")
    plt.savefig(f"./results/{model_name}/{category}_{num_epochs}_soft_doc_acc.png")
    plt.show()
    
    with open(f"./results/{model_name}/{category}_{num_epochs}.json", 'w') as fp:
            json.dump(results, fp)
            
    torch.save(model, f"./models/{model_name}/{category}_{num_epochs}.pt")
    

def evaluation_bert(model, test_loader, criterion, device):
    model.eval()
    valid_running_loss = 0.0
    with torch.no_grad():                    
        # validation loop
        document_eval_outputs = []
        document_eval_labels = []
        eval_outputs = []
        eval_labels = []
        for _, eval_sample in enumerate(test_loader):
            
            labels = torch.FloatTensor(eval_sample["labels"]).to(device).unsqueeze(0)
            inputs = torch.LongTensor(eval_sample["subparts"]).to(device)

            outputs = []

            for inp in inputs:
                output = torch.sigmoid(model(inp.unsqueeze(0)).logits)
                outputs.append(output)
            
            eval_outputs.extend(outputs)
            eval_labels.extend(labels)

            outputs = torch.cat(outputs)
            
            outputs = outputs.squeeze(0)
            labels = labels.squeeze(0)
            
            if len(outputs) > 1:
                outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, labels)
            
            valid_running_loss += loss.item()

            document_eval_outputs.append(outputs)
            document_eval_labels.append(labels)
        eval_outputs = torch.cat(eval_outputs)
        eval_labels = torch.cat(eval_labels)

        print()
        print("---------- EVALUATION ----------")
        print("Document Accuracy:", calculate_document_accuracy(document_eval_outputs, document_eval_labels))
        print("Soft Document Accuracy:", calculate_soft_document_accuracy(document_eval_outputs, document_eval_labels))
        print("Chunk Accuracy:", calculate_chunk_accuracy(eval_outputs.squeeze(), eval_labels))

        print("Loss:", valid_running_loss/len(test_loader))
        metrics = get_metrics(calculate_metrics(eval_outputs, eval_labels))
        for k, v in metrics.items():
            print(f"{k}:", v)
        
        return valid_running_loss/len(test_loader), calculate_soft_document_accuracy(document_eval_outputs, document_eval_labels), metrics, calculate_chunk_accuracy(eval_outputs.squeeze(), eval_labels), calculate_document_accuracy(document_eval_outputs, document_eval_labels)