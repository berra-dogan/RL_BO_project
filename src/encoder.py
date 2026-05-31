import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the attention layer for the encoder
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(K.size(-1), dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

# Define the DeepSet encoder
class DeepSetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepSetEncoder, self).__init__()
        self.attention = AttentionLayer(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.attention(x)
        x = F.relu(self.fc1(x))
        x = x.mean(dim=0)
        x = self.fc2(x)
        return x
