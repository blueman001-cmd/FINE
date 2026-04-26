import torch
import time
import warnings
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from utils.poolers import Pooling
from utils.utils import set_random_seed
import numpy as np
from model.eval import batch_level_evaluation, evaluate_entity_level_using_knn
from utils.config import build_args
warnings.filterwarnings('ignore')


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
    print(dataset_name)
    print(max_epoch)
    print(num_hidden)
    print(num_layer)
    print(n)
    print(p)
    set_random_seed(0)
    if dataset_name == 'streamspot' or dataset_name == 'wget':
        dataset = load_batch_level_dataset(dataset_name)
        n_node_feat = dataset['n_feat']
        n_edge_feat = dataset['e_feat']
        print('开始构建模型')
        model = build_model(n_node_feat,num_hidden,n_edge_feat,num_layer,n,p)
        print('模型构建完毕')
        model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset_name), map_location=device))
        model = model.to(device)
        pooler = Pooling(main_args.pooling)
        test_auc, test_std = batch_level_evaluation(model, pooler, device, ['knn'], args.dataset, n_node_feat,
                                                    n_edge_feat)
    else:
        metadata = load_metadata(dataset_name)
        n_node_feat = metadata['node_feature_dim']
        n_edge_feat = metadata['edge_feature_dim']
        model = build_model(n_node_feat,num_hidden,n_edge_feat,num_layer,n,p)
        model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset_name), map_location=device))
        model = model.to(device)
        model.eval()
        malicious, _ = metadata['malicious']
        n_train = metadata['n_train']
        n_test = metadata['n_test']

        with torch.no_grad():
            x_train = []
            for i in range(n_train):
                g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
                g.ndata['h'] = g.ndata.pop('attr')
                g.edata['h'] = g.edata.pop('attr')
                x_train.append(model.embed(g).cpu().numpy())
                del g
            x_train = np.concatenate(x_train, axis=0)
            skip_benign = 0
            x_test = []
            for i in range(n_test):
                g = load_entity_level_dataset(dataset_name, 'test', i).to(device)
                g.ndata['h'] = g.ndata.pop('attr')
                g.edata['h'] = g.edata.pop('attr')
                # Exclude training samples from the test set
                if i != n_test - 1:
                    skip_benign += g.number_of_nodes()
                x_test.append(model.embed(g).cpu().numpy())
                del g
            x_test = np.concatenate(x_test, axis=0)

            n = x_test.shape[0]
            y_test = np.zeros(n)
            y_test[malicious] = 1.0
            malicious_dict = {}
            # for i, m in enumerate(malicious):
            #     malicious_dict[m] = i

            # Exclude training samples from the test set
            test_idx = []
            for i in range(x_test.shape[0]):
                if i >= skip_benign or y_test[i] == 1.0:
                    test_idx.append(i)
            result_x_test = x_test[test_idx]
            result_y_test = y_test[test_idx]
            del x_test, y_test
            test_auc, test_std, _, _ = evaluate_entity_level_using_knn(dataset_name, x_train, result_x_test,
                                                                       result_y_test)
    print()
    end_time = time.time()
    elapsed = end_time - start_time
    # 转为分钟和秒
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"程序运行时间：{minutes} 分 {seconds:.2f} 秒")
    return


if __name__ == '__main__':
    start_time = time.time()
    args = build_args()
    main(args)
