from __future__ import print_function
import os
from os.path import isfile, join, isdir
import random
import numpy as np
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from VidT_lite import VidT

def shuffleData(X, y):
	assert len(X) == len(y)
	'''p = np.random.permutation(len(X))
	return X[p], y[p]'''
	temp = list(zip(X.copy(), y.copy()))
	random.shuffle(temp)
	X_shuffled, y_shuffled = zip(*temp)
	return X_shuffled, y_shuffled

pkg_dir = 'saved_models'
train_data = np.load(join('data', 'patch_data', '40', 'all_data.npy'), allow_pickle=True)
train_indices_unshuffled = np.load(join('data', 'patch_data', '40', 'all_indices.npy'), allow_pickle=True)
train_labels_unshuffled = np.load(join('data', 'patch_data', '40', 'all_labels.npy'), allow_pickle=True)
train_indices, train_labels = shuffleData(train_indices_unshuffled, train_labels_unshuffled)

# pkg_dir = 'saved_models'
# train_data = np.load(join('data', 'patch_data', '40', 'all_data.npy'), allow_pickle=True)[:100,...]
# train_indices_unshuffled = np.load(join('data', 'patch_data', '40', 'all_indices.npy'), allow_pickle=True)[:100,...]
# train_labels_unshuffled = np.load(join('data', 'patch_data', '40', 'all_labels.npy'), allow_pickle=True)[:100,...]
# train_indices, train_labels = shuffleData(train_indices_unshuffled, train_labels_unshuffled)

# val_data = np.concatenate([np.load(join('data', 'patch_clips', '40', 'validation_validation_0000_data.npy'), allow_pickle=True), np.load(join('data', 'patch_clips', '40', 'validation_validation_0001_data.npy'), allow_pickle=True), np.load(join('data', 'patch_clips', '40', 'validation_validation_0002_data.npy'), allow_pickle=True)], axis = 0)
# val_labels = np.concatenate([np.load(join('data', 'patch_data', '40', 'validation_validation_0000_labels.npy'), allow_pickle=True), np.load(join('data', 'patch_data', '40', 'validation_validation_0001_labels.npy'), allow_pickle=True), np.load(join('data', 'patch_data', '40', 'validation_validation_0002_labels.npy'), allow_pickle=True)], axis = 0)[:, :2]

val_data = np.concatenate([np.load(join('data', 'patch_small_dataset', 'sdata1.npy'), allow_pickle=True), np.load(join('data', 'patch_small_dataset', 'sdata2.npy'), allow_pickle=True)], axis = 0)
val_labels = np.concatenate([np.load(join('data', 'patch_small_dataset', 'slabel1.npy'), allow_pickle=True), np.load(join('data', 'patch_small_dataset', 'slabel2.npy'), allow_pickle=True)], axis = 0)[:, :2]

print("len(train_data)= {}, \n len(train_labels)={}, \n val_data.shape={}, \n  val_labels.shape={}".format(len(train_data), len(train_labels), val_data.shape, val_labels.shape))


parser = argparse.ArgumentParser(description='PyTorch Video Transformer Model')
parser.add_argument('--cuda', type=int, default=0, help='use CUDA')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.00007)
parser.add_argument('--gamma', type=float, default=0.7)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_frames', type=int, default=10)
parser.add_argument('--num_dims', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--dim_head', type=int, default=128)
parser.add_argument('--mlp_dim', type=int, default=128)
parser.add_argument('--drop_prob', type=float, default=0.4)
parser.add_argument('--emb_drop_prob', type=float, default=0.4)
parser.add_argument('--cls_dim', type=int, default=64)
args = parser.parse_args()
args_string = str(args) + '\n'

batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
gamma = args.gamma
seed = args.seed
num_frames = args.num_frames
num_dims = args.num_dims
num_layers = args.num_layers
num_heads = args.num_heads
dim_head = args.dim_head
mlp_dim = args.mlp_dim
drop_prob = args.drop_prob
emb_drop_prob = args.emb_drop_prob
cls_dim = args.cls_dim
device = torch.device('cuda:{}'.format(args.cuda))

# trail Training settings
# batch_size = 20
# epochs = 2
# lr = 0.00007
# gamma = 0.7
# seed = 42
# num_frames = 10
# num_dims = 20
# num_layers = 2
# num_heads = 2
# dim_head = 10
# mlp_dim = 10
# drop_prob = 0.4
# emb_drop_prob = 0.4
# cls_dim = 10

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

#print(device)
model = VidT(
    num_patches = val_data.shape[1]//num_frames,
    num_acc = val_labels[0].shape[0],
    num_frames = num_frames,
    patch_dim = val_data.shape[-1], #flatten patch_dim xp
    dim=num_dims,
    depth = num_layers,
    heads = num_heads,
    mlp_dim = mlp_dim,
    cls_dim=cls_dim,
    pool = 'cls',
    dim_head = dim_head,
    dropout = drop_prob,
    emb_dropout = emb_drop_prob,
    device = device)
    
model = model.to(device)
print("# of parameters = ", sum(p.numel() for p in model.parameters() if p.requires_grad))

class VidTDataset(Dataset):
    def __init__(self, clips, labels):
        self.clips = clips
        self.acc = labels

    def __len__(self):
        self.filelength = len(self.acc)
        return self.filelength

    def __getitem__(self, idx):
        clip = self.clips[idx]
        acc = self.acc[idx]
        return clip, acc


valid_dataset = VidTDataset(val_data, val_labels)

valid_loader = DataLoader(dataset = valid_dataset, batch_size=batch_size)

# loss function
criterion = nn.L1Loss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

def get_batch_data(all_indices, all_data, start, end):
	batch_idxs = all_indices[start:end]
	batch_data = []
	for i in batch_idxs:
		batch_data.append(np.expand_dims(np.concatenate(all_data[i-10:i], axis = 0).squeeze(),0))
	return np.concatenate(batch_data, axis=0)
	
now =  datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
open('output_' + now + '.txt', 'w').close()
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_mae = np.zeros(val_labels[0].shape[0])
    start_idx = 0
    end_idx = np.min([start_idx + batch_size, len(train_labels)])
    num_batches = np.ceil(len(train_labels)/batch_size)
    while start_idx < len(train_labels):
        data = get_batch_data(train_indices, train_data, start_idx, end_idx)
        label = np.array(train_labels[start_idx:end_idx])
        data = torch.tensor(data).to(device)
        label = torch.tensor(label[:, :2]).to(device).float()#ignore z
        output = model(data)

        loss = criterion(output, label).float()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sae = torch.sum(torch.abs(output - label).float(), axis = 0).detach().cpu().numpy()
        epoch_mae += sae
        epoch_loss += loss/num_batches
        start_idx = np.min([start_idx + batch_size, len(train_labels)])
        end_idx = np.min([start_idx + batch_size, len(train_labels)])

    with torch.no_grad():
        model.eval()
        epoch_val_mae = np.zeros(val_labels[0].shape[0])
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            sae = torch.sum(torch.abs(val_output - label).float(), axis = 0).detach().cpu().numpy()
            epoch_val_mae += sae
            epoch_val_loss += val_loss / len(valid_loader)
    epoch_mae = epoch_mae / len(train_labels)
    epoch_val_mae = epoch_val_mae / len(val_labels)
    output_str = f"Epoch : {epoch+1} - training_loss : {epoch_loss:.4f} - training_mae: {epoch_mae} - val_loss : {epoch_val_loss:.4f} - val_mae: {epoch_val_mae}\n"
    print(
        output_str
    )
    with open('output_' + now + '.txt', "a") as myfile:
        if epoch == 0:
            myfile.write(args_string)
        myfile.write(output_str)
    state = {
        "epoch":
        epoch + 1,
        # save model
        "model":
        model.module.state_dict()
        if hasattr(model, "module") else model.state_dict(),
        "optim":
        optimizer.state_dict(),
    }
    save_path = os.path.join(pkg_dir, 'vidt', str(now)+'__'+str(epoch+1)+'.pkg')
    torch.save(state, save_path)

