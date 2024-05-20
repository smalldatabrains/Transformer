import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        x = x + self.encoding[:, :x.size(1), :].to(x.device)
        return x

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, max_len=5000):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len)
        encoder_layers = nn.TransformerEncoderLayer(model_dim, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.model_dim)
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        output = self.fc_out(output)
        return output

# Sample Dataset
class SampleDataset(Dataset):
    def __init__(self, num_samples, seq_length, vocab_size, num_classes):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length))
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Training Function
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for src, labels in dataloader:
            optimizer.zero_grad()
            output = model(src)
            # We only need the output corresponding to the last token
            output = output[:, -1, :] 
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}')

# Parameters
input_dim = 1000  # Vocabulary size
model_dim = 512   # Embedding size
num_heads = 8     # Number of heads in multi-head attention
num_layers = 6    # Number of transformer layers
output_dim = 10   # Number of output classes
max_len = 500     # Maximum length of input sequence
batch_size = 32   # Batch size for training
num_epochs = 10   # Number of epochs for training

# Initialize model, dataset, dataloader, loss function, and optimizer
model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim, max_len)
dataset = SampleDataset(1000, 100, input_dim, output_dim)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, dataloader, criterion, optimizer, num_epochs)
