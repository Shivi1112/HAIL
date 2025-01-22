#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# model1 = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Function to generate queries
def query_generator(col, row, bert):
    query = []
    for c in col:
        if not pd.isna(row[c]):
            text = row[c]
            tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                outputs = bert(**tokens)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[:, 0, :]  # Shape: [1, 768]

            # Reduce dimension to 128 right after getting BERT embedding
            linear_reduction = nn.Linear(768, 128)
            reduced_output = linear_reduction(pooled_output)
            query.append(reduced_output)
    return query

# Function to generate keys and values
def key_generator(context_vector):
    linear_layer = nn.Linear(128, 128)
    key = linear_layer(context_vector)
    value = linear_layer(context_vector)
    return key, value

def create_embedding(text, tokenizer, model):
    inputs = tokenizer.encode_plus(
        text,
        padding=True,
        max_length=512,
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state
    # First reduce to 128 dimensions
    linear_reduction = nn.Linear(768, 128)
    reduced_hidden = linear_reduction(last_hidden_state)

    embedding = torch.nn.functional.normalize(torch.mean(reduced_hidden, dim=1), p=2, dim=1).numpy()
    return embedding


class BERTGCNModel(nn.Module):
    def __init__(self, num_classes):
        super(BERTGCNModel, self).__init__()
        # Modified for 128-dimensional inputs
        self.conv1 = GCNConv(1024, 256)  # 8 columns * 128 dimensions = 1024
        self.conv2 = GCNConv(256, 128)
        self.fc = nn.Linear(1152, num_classes)  # 1024 + 128 (GCN)

    def forward(self, bert_output, edge_index, edge_weight):
        gcn_output = F.relu(self.conv1(bert_output, edge_index=edge_index, edge_weight=edge_weight))
        gcn_output = F.relu(self.conv2(gcn_output, edge_index=edge_index, edge_weight=edge_weight))
        concatenated_embedding = torch.cat((bert_output, gcn_output), dim=1)
        logits = self.fc(concatenated_embedding)
        return torch.sigmoid(logits)

def create_embedding(text, tokenizer, model):
    inputs = tokenizer.encode_plus(
        text,
        padding=True,
        max_length=512,
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state

        # First reduce to 128 dimensions
        linear_reduction = nn.Linear(768, 128)
        reduced_hidden = linear_reduction(last_hidden_state)

        # Use detach() before converting to numpy
        embedding = torch.nn.functional.normalize(
            torch.mean(reduced_hidden, dim=1),
            p=2,
            dim=1
        ).detach().numpy()

    return embedding

class ClinicalNotesDataset(Dataset):
    def __init__(self, data, model, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.bert_model = model
        self.columns = ['CHIEF_COMPLAINT', 'PRESENT_ILLNESS', 'MEDICAL_HISTORY', 'MEDICATION_ADM',
                       'ALLERGIES', 'PHYSICAL_EXAM', 'FAMILY_HISTORY', 'SOCIAL_HISTORY']
        # Initialize dimension reduction layer
        self.dim_reduction = nn.Linear(768, 128)

    def __len__(self):
        return len(self.data)
    
    def _process_text(self, text, row):
        if pd.isna(text):
            query = query_generator(self.columns, row, self.bert_model)
            # Initialize with 128 dimensions directly
            key, value = torch.randn((1, 128)), torch.randn((1, 128))

            with torch.no_grad():
                for i in query:
                    attention_scores = torch.matmul(i, key.transpose(0, 1))
                    attention_weights = torch.softmax(attention_scores, dim=-1)
                    context_vector = torch.matmul(attention_weights, value).squeeze(0)
                    key, value = key_generator(context_vector)
                    key, value = key.unsqueeze(0), value.unsqueeze(0)

                embedding = context_vector.unsqueeze(0)
        else:
            words = nltk.word_tokenize(text)
            chunk_size = 512
            chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

            with torch.no_grad():
                all_embeddings = [create_embedding(' '.join(chunk), self.tokenizer, self.bert_model)
                                for chunk in chunks]
                # all_embeddings are already 128-dimensional from create_embedding
                embedding = torch.nn.functional.normalize(
                    torch.sum(torch.tensor(all_embeddings), dim=0),
                    p=2,
                    dim=1
                ).detach().numpy()

        return embedding

    def _get_embedding(self, row):
        with torch.no_grad():
            embeddings = [self._process_text(row[col], row) for col in self.columns]
        return embeddings

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        embeddings = self._get_embedding(row)

        id_ = row['id']
        label = row['hospital_expire_flag']

        embeddings_list = [torch.tensor(embedding_array, dtype=torch.float32)
                          for embedding_array in embeddings]
        concatenated_embeddings = torch.cat(embeddings_list, dim=1)

        return {
            'id': torch.tensor(id_, dtype=torch.long),
            'embeddings': concatenated_embeddings,
            'label': torch.tensor(label, dtype=torch.long)
        }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[18]:


def create_dataloader(data, model, tokenizer,batch_size):
    dataset = ClinicalNotesDataset(data,model,tokenizer)
#     print('ds',dataset.shape)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    return dataloader



import torch

def calculate_cosine_similarity(embedding1, embedding2):
    
    dot_product = torch.dot(embedding1, embedding2)
    norm1 = torch.norm(embedding1)
    norm2 = torch.norm(embedding2)
    cosine_similarity = dot_product / (norm1 * norm2)
    return cosine_similarity.item()  # Return as float


# In[21]:


# Define your model

class BERTGCNModel(nn.Module):
    def __init__(self, num_classes): #======================================
        super(BERTGCNModel, self).__init__()
        # Define GCN layers
        self.conv1 = GCNConv(1024, 512)
        self.conv2 = GCNConv(512, 128)
        self.fc = nn.Linear(128, num_classes)  # 768 (BERT) + 128 (GCN)
       
    def forward(self, bert_output,edge_index,edge_weight):
        gcn_output = F.relu(self.conv1(bert_output, edge_index=edge_index, edge_weight=edge_weight))
        gcn_output = F.relu(self.conv2(gcn_output, edge_index=edge_index, edge_weight=edge_weight))
#         concatenated_embedding = torch.cat((bert_output, gcn_output), dim=1)

        # Pass through dense layer
        logits = self.fc(gcn_output)

        return torch.sigmoid(logits)


# In[22]:


def create_graph(node_features, num_nodes):
    edge_index = []
    edge_weight = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Extract embeddings for nodes i and j
#                 print('nf',node_features.shape)
            embedding_i = node_features[i].squeeze()  # Remove singleton dimension
#                 print('i',i,embedding_i.shape)
            embedding_j = node_features[j].squeeze()  # Remove singleton dimension
#                 print('j',j,embedding_j.shape)
            # Compute cosine similarity between the embeddings
            cosine_similarity = calculate_cosine_similarity(embedding_i, embedding_j)
            edge_index.append([i, j])
            if cosine_similarity>0:
                edge_weight.append(cosine_similarity)
            else:
                cosine_similarity=0
                edge_weight.append(cosine_similarity)
#             print(edge_index,'+++',edge_weight)
    return edge_index, edge_weight


# In[23]:


# model = BERTGCNModel(num_classes=1).to(device)  # Move model to CUDA device

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Move tokenizer to CUDA device
# bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)  # Move BERT model to CUDA device

# Define DataLoader and move it to CUDA if applicable
# train_dataset.to(device)


# In[24]:


from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import numpy as np

def calculate_class_weights(labels):
    """Calculate class weights for imbalanced dataset"""
    class_counts = np.bincount(labels)
    total = len(labels)
    class_weights = torch.FloatTensor([total / (len(class_counts) * c) for c in class_counts])
    return class_weights.to(device)

def evaluate_metrics(y_true, y_pred, y_prob):
    """Calculate evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auroc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob)
    }
    return metrics


# In[30]:


def evaluate(data, embedding_model, tokenizer, model):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        cs = create_dataloader(data, embedding_model, tokenizer, batch_size=16)
        for batch in cs:
            bert_output = batch['embeddings'].squeeze().to(device)
            labels = batch['label'].to(device)

            num_nodes = bert_output.shape[0]
            edge_index, edge_weight = create_graph(bert_output, num_nodes)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
            edge_weight = torch.tensor(edge_weight, dtype=torch.float).to(device)

            logits = model(bert_output, edge_index, edge_weight)
            probs = torch.sigmoid(logits).squeeze()
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = evaluate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    return metrics


# In[31]:


def training(train_df, val_df, embedding_model, tokenizer, model, epochs, patience=5):
    # Calculate class weights from training data
    train_labels = train_df['hospital_expire_flag'].values
    class_weights = calculate_class_weights(train_labels)

    # Define criterion with weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

    # Define optimizer
    optimizer = optim.Adam([
        {'params': embedding_model.parameters()},
        {'params': model.parameters()}
    ], lr=0.0001)

    # Training history
    history = {
        'train_loss': [],
        'val_metrics': []
    }

    # Early stopping variables
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        cs = create_dataloader(train_df, embedding_model, tokenizer, batch_size=16)

        for batch in tqdm(cs, desc=f'Epoch {epoch+1}'):
            bert_output = batch['embeddings'].squeeze().to(device)
            label = batch['label'].to(device)

            num_nodes = bert_output.shape[0]
            edge_index, edge_weight = create_graph(bert_output, num_nodes)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
            edge_weight = torch.tensor(edge_weight, dtype=torch.float).to(device)

            optimizer.zero_grad()
            logits = model(bert_output, edge_index, edge_weight)
            loss = criterion(logits.squeeze(), label.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * bert_output.size(0)

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_df)
        val_metrics = evaluate(val_df, embedding_model, tokenizer, model)
        val_loss = val_metrics.get('loss', float('inf'))  # Assuming 'loss' is in val_metrics

        # Save history
        history['train_loss'].append(epoch_loss)
        history['val_metrics'].append(val_metrics)

        # Print metrics
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {epoch_loss:.4f}")
        print(f"Validation Metrics:")
        for metric, value in val_metrics.items():
            print(f"{metric}: {value:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save best model state
            best_model_state = {
                'embedding_model': embedding_model.state_dict(),
                'model': model.state_dict(),
                'epoch': epoch,
                'val_metrics': val_metrics
            }
        else:
            counter += 1
            print(f"Early stopping counter: {counter}/{patience}")

        if counter >= patience:
            print(f"\nEarly stopping triggered after epoch {epoch+1}")
            # Restore best model state
            embedding_model.load_state_dict(best_model_state['embedding_model'])
            model.load_state_dict(best_model_state['model'])
            break

    # Print best performance
    print("\nBest model performance:")
    print(f"Epoch: {best_model_state['epoch'] + 1}")
    print("Validation Metrics:")
    for metric, value in best_model_state['val_metrics'].items():
        print(f"{metric}: {value:.4f}")

    return history, best_model_state


# Usage
def main():
    # Load your data
    train_df=pd.read_csv('mimic/mimic_col_test.csv')
    val_df=pd.read_csv('mimic/mimic_col_test.csv')
    test_df=pd.read_csv('mimic/mimic_col_test.csv')
    print(device)
    # Initialize your models
    model = BERTGCNModel(num_classes=1).to(device)  # Move model to CUDA device

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Move tokenizer to CUDA device
    bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)  # Move BERT model to CUDA device

    history,mod = training(train_df, val_df, bert_model, tokenizer, model, epochs=5)
    # Plot training results
    plt.figure(figsize=(15, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()

    # Plot validation metrics
    plt.subplot(1, 2, 2)
    metrics = ['accuracy', 'f1', 'auroc', 'auprc']
    for metric in metrics:
        values = [m[metric] for m in history['val_metrics']]
        plt.plot(values, label=f'Validation {metric}')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics Over Time')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

    # Train and evaluate
    test_metrics = evaluate(test_df,bert_model, tokenizer, model)
    print(test_metrics)
    # Print summary
    print("\nResults saved to 'model_metrics.csv'")
    print("Plots saved to 'complete_training_results.png'")

    
if __name__ == "__main__":
    main()


