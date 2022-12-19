from torch.utils.data import Dataset

class DocumentData:

    def __init__(self, subparts, labels, filename):
        self.subparts = subparts
        self.labels = labels
        self.filename = filename

class BinaryCUADDataset(Dataset):
    def __init__(self, documents):
        self.documents = documents

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        document = self.documents[idx]

        filename = document.filename
        subparts = document.subparts
        labels = document.labels
        
        sample = {"labels": labels, "subparts": subparts}
        
        return sample