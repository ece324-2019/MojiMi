import torch.utils.data as data

# Creating dataset for getting inputs

class vggDataset(data.Dataset):

    def __init__(self, X, y):
        self.X = X                  # the feautres input
        self.y = y                  # the label

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        features = self.X[index]
        label = self.y[index-1]
        return features, label