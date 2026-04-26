import os
import random
import torch
import warnings
from tqdm import tqdm
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from model.train import batch_level_train
from utils.utils import set_random_seed, create_optimizer
from utils.config import build_args
warnings.filterwarnings('ignore')


def extract_dataloaders(entries, batch_size):
    random.shuffle(entries)
    train_idx = torch.arange(len(entries))
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = GraphDataLoader(entries, batch_size=batch_size, sampler=train_sampler)
    return train_loader


def main(main_args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    dataset_name = main_args.dataset
    if dataset_name == 'streamspot':
        max_epoch = 3
        num_hidden = 128
        num_layer = 1
        n = 1
        p = 100
    elif dataset_name == 'wget':   
        max_epoch = 2
        num_hidden = 256
        num_layer = 4
        n = 0
        p = 1
    elif dataset_name == 'cadets':   
        max_epoch = 7
        num_hidden = 64
        num_layer = 3
        n = 0
        p = 1
    elif dataset_name == 'theia':   
        max_epoch = 3
        num_hidden = 64
        num_layer = 3
        n = 0
        p = 1
    else:
        max_epoch = 8
        num_hidden = 64
        num_layer = 3
        n = 0
        p = 1
        
    set_random_seed(0)
    print(dataset_name)
    print(max_epoch)
    print(num_hidden)
    print(num_layer)
    print(n)
    print(p)
    if dataset_name == 'streamspot' or dataset_name == 'wget':
        if dataset_name == 'streamspot':
            batch_size = 12
        else:
            batch_size = 1
        dataset = load_batch_level_dataset(dataset_name)
        n_node_feat = dataset['n_feat']
        n_edge_feat = dataset['e_feat']
        graphs = dataset['dataset']
        train_index = dataset['train_index']
        print('开始构建模型')
        model = build_model(n_node_feat,num_hidden,n_edge_feat,num_layer,n,p)
        print('模型构建完毕')
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=0.)
        model = batch_level_train(model, graphs, (extract_dataloaders(train_index, batch_size)),
                                  optimizer, max_epoch, device, n_node_feat, n_edge_feat)
        # torch.save(model.state_dict(), "./checkpoints/checkpoint-{}.pt".format(dataset_name))
    else:
        metadata = load_metadata(dataset_name)
        n_node_feat = metadata['node_feature_dim']
        n_edge_feat = metadata['edge_feature_dim']
        model = build_model(n_node_feat,num_hidden,n_edge_feat,num_layer,n,p)
        model = model.to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=0.)
        epoch_iter = tqdm(range(max_epoch))
        n_train = metadata['n_train']
        for epoch in epoch_iter:
            epoch_loss = 0.0
            for i in range(n_train):
                g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
                g.ndata['h'] = g.ndata.pop('attr')
                g.edata['h'] = g.edata.pop('attr')
                model.train()
                loss = model(g)
                loss /= n_train
                optimizer.zero_grad()
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                del g
            epoch_iter.set_description(f"Epoch {epoch+1} | train_loss: {epoch_loss:.4f}")
        # torch.save(model.state_dict(), "./checkpoints/checkpoint-{}.pt".format(dataset_name))
        # save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset_name)
        # if os.path.exists(save_dict_path):
            # os.unlink(save_dict_path)
    return


if __name__ == '__main__':
    args = build_args()
    main(args)
