import torch
import torch.nn as nn

class MGNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(MGNNLayer, self).__init__()
        self.W1 = nn.Linear(in_features, out_features, bias=False)
        self.W2 = nn.Linear(in_features, out_features, bias=True)
        self.activation = nn.ReLU()

    def forward(self, H, A):
        # H : tenseur de dimension (batch_size, n, in_features)
        # A : matrice d'adjacence (batch_size, n, n)
        
        # Agrégation du voisinage : A * H * W1
        neighbors_aggr = torch.bmm(A, self.W1(H)) 
        
        # Transformation du noeud lui-même : H * W2
        self_feat = self.W2(H)
        
        return self.activation(neighbors_aggr + self_feat)
    
class MGNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers):
        super(MGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(MGNNLayer(in_features, hidden_features))
        for _ in range(num_layers - 2):
            self.layers.append(MGNNLayer(hidden_features, hidden_features))
        self.layers.append(MGNNLayer(hidden_features, out_features))

    def forward(self, H, A):
        for layer in self.layers:
            H = layer(H, A)
        return H # Sortie de dimension (batch, n, out_features)