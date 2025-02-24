import torch
import torch.nn as nn


class NeuralCollaborativeFilter(nn.Module):
    """
        Neural Collaborative Filtering\n
        Inputs: num_users, num_items, embedding_dim=100
    """
    def __init__(self,num_users,num_items,embedding_dim=100):
        
        super(NeuralCollaborativeFilter,self).__init__()

        self.user_embedding = nn.Embedding(num_users,embedding_dim)
        self.item_embedding = nn.Embedding(num_items,embedding_dim)

        self.network = nn.Sequential(
            nn.Linear(2*embedding_dim,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self,user_input,item_input):
        x = self.user_embedding(user_input)
        y = self.user_embedding(item_input)

        input_vector = torch.cat([x,y],dim=1)
        preds = self.network(input_vector)
        return preds.squeeze()

