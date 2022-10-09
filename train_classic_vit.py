from dataloader import ESC_Dataset
import torch
from models.vit import ViT
import os
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.nn as nn



full_dataset = np.load(os.path.join("esc-50", "esc-50-data.npy"), allow_pickle = True)


train_dataset = ESC_Dataset(dataset = full_dataset, esc_fold=0, eval_mode = False)
eval_dataset = ESC_Dataset(dataset = full_dataset, esc_fold=0, eval_mode = True)



train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4, collate_fn=None, pin_memory=False)
eval_dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=False, num_workers=4, collate_fn=None, pin_memory=False)



model = ViT(
    image_size = (320,128),
    patch_size = (40,16),
    channels = 1,
    num_classes = 50,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.0,
    emb_dropout = 0.0
)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = model.to(device)


def eval_model(model, eval_dataset, device):
    model.eval()
    forecast, true_labs = [], []
    with torch.no_grad():
        for data, labs in tqdm(eval_dataset):
            data, labs = data.to(device), labs[:,0].cpu()
            true_labs.append(labs)
            outputs = model(data)
            
            outputs = outputs.detach().cpu().numpy().argmax(axis=1)
            forecast.append(outputs)
    forecast = [x for sublist in forecast for x in sublist]
    true_labs = [x for sublist in true_labs for x in sublist]
    return f1_score(forecast, true_labs, average='macro'), accuracy_score(forecast, true_labs)



criterion = nn.CrossEntropyLoss()




n_epoch = 100
best_f1 = 0
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for epoch in range(n_epoch):
    model.train()
    for data, labs in tqdm(train_dataloader):
        data, labs = data.to(device), labs.to(device)[:,0]
        outputs = model(data)
        loss = criterion(outputs, labs) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # del outputs
        # torch.cuda.empty_cache( )
#     if epoch % 10 == 0:
    f1, accuracy = eval_model(model, eval_dataloader, device)
    f1_train, accuracy_train = eval_model(model, train_dataloader, device)
    print(f'epoch: {epoch}, f1_test: {f1}, accuracy_test: {accuracy}, f1_train: {f1_train},  accuracy_train: {accuracy_train}')
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), 'classic_vit.pt')

    lr = lr * 0.95
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr