import torch
import torch.nn as nn
from model import NeuralCollaborativeFilter

def get_recommendations(model,user_id,num_recommendations,device):
    """
    Return recommendations for a user id\n
    Input : model,user_id,num_recommendations,device
    """
    model.eval()
    recommendations = []
    with torch.no_grad():
        user_index = user_mapping[user_id]
        user_input = torch.tensor([user_index],dtype=torch.long,device=device)
        items_list = list(item_mapping.values())
        items_input = torch.tensor(items_list,dtype=torch.long,device=device)

        user_input = user_input.expand(len(items_input))

        preds = model(user_input,items_input)

        top_recommendations = torch.topk(preds,num_recommendations)
        reverse_item_mapping = {v: k for k, v in item_mapping.items()}
        
        for i in top_recommendations.indices:
            recommendations.append(reverse_item_mapping[i.item()])
    
    return recommendations

def evaluate_model(model,test_dataloader,device):
    model.eval()
    criterion = nn.MSELoss()
    batch_loss = 0
    with torch.no_grad():
        for users,items,ratings in test_dataloader:
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device)
            preds = model(users,items)
            loss = criterion(preds,ratings)
            batch_loss += loss.item()
    return batch_loss/len(test_dataloader)

def inference(user_id,num_user,num_items,weight_path,device):
    model = NeuralCollaborativeFilter(num_users=num_user,num_items=num_items)
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    recommendations = get_recommendations(model,user_id,num_recommendations=5,device=device)
    print("----- Recommended Pratilipis -----\n")
    for r in recommendations:
        print(f"ID:{r}")