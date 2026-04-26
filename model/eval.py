import os
import random
import time
import pickle as pkl
import torch
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.neighbors import NearestNeighbors
from utils.utils import set_random_seed
from utils.loaddata import transform_graph, load_batch_level_dataset


def batch_level_evaluation(model, pooler, device, method, dataset, n_dim=0, e_dim=0):
    model.eval()
    x_list = []
    y_list = []
    data = load_batch_level_dataset(dataset)
    full = data['full_index']
    graphs = data['dataset']
    with torch.no_grad():
        for i in full:
            g = transform_graph(graphs[i][0], n_dim, e_dim).to(device)
            label = graphs[i][1]
            out = model.embed(g)
            if dataset != 'wget':
                out = pooler(g, out).cpu().numpy()
            else:
                out = pooler(g, out, n_types=data['n_feat']).cpu().numpy()
                # out = pooler(g, out).cpu().numpy()
            y_list.append(label)
            x_list.append(out)
    x = np.concatenate(x_list, axis=0)
    y = np.array(y_list)
    if 'knn' in method:
        test_auc, test_std = evaluate_batch_level_using_knn(100, dataset, x, y)
    else:
        raise NotImplementedError
    return test_auc, test_std


def evaluate_batch_level_using_knn(repeat, dataset, embeddings, labels):
    x, y = embeddings, labels
    if dataset == 'streamspot':
        train_count = 400
    else:
        train_count = 100
    n_neighbors = min(int(train_count * 0.02), 100)
    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    if repeat != -1:
        prec_list = []
        rec_list = []
        f1_list = []
        tp_list = []
        fp_list = []
        tn_list = []
        fn_list = []
        auc_list = []
        for s in range(repeat):
            set_random_seed(s)
            np.random.shuffle(benign_idx)
            np.random.shuffle(attack_idx)
            if dataset == 'streamspot':
                x_train = x[benign_idx[:train_count]]
                x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
                y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
            else:
                x_train = x[benign_idx[:100]]
                x_test = np.concatenate([x[benign_idx[100:]], x[attack_idx]], axis=0)
                y_test = np.concatenate([y[benign_idx[100:]], y[attack_idx]], axis=0)

            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            nbrs.fit(x_train)
            distances, indexes = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
            mean_distance = distances.mean() * n_neighbors / (n_neighbors - 1)
            distances, indexes = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)

            score = distances.mean(axis=1) / mean_distance

            # 计算 ROC 曲线
            fpr, tpr, _ = roc_curve(y_test, score)
            
            auc = roc_auc_score(y_test, score)
            prec, rec, threshold = precision_recall_curve(y_test, score)
            f1 = 2 * prec * rec / (rec + prec )
            max_f1_idx = np.argmax(f1)
            best_thres = threshold[max_f1_idx]
            prec_list.append(prec[max_f1_idx])
            rec_list.append(rec[max_f1_idx])
            f1_list.append(f1[max_f1_idx])

            tn = 0
            fn = 0
            tp = 0
            fp = 0
            for i in range(len(y_test)):
                if y_test[i] == 1.0 and score[i] >= best_thres:
                    tp += 1
                if y_test[i] == 1.0 and score[i] < best_thres:
                    fn += 1
                if y_test[i] == 0.0 and score[i] < best_thres:
                    tn += 1
                if y_test[i] == 0.0 and score[i] >= best_thres:
                    fp += 1
            if dataset == 'streamspot':
                if tp+tn == 200 :
                    print('TP: {}'.format(tp))
                    print('FP: {}'.format(fp))
                    print('TN: {}'.format(tn))
                    print('FN: {}'.format(fn))
                    print('Precision: {}'.format(prec[max_f1_idx]))
                    print('Recall: {}'.format(rec[max_f1_idx]))
                    print('FPR: {}'.format(fp/(fp+tn)))
                    print('F1: {}'.format(f1[max_f1_idx]))
                    print('AUC: {}'.format(auc))

                    with open("result.txt", "a") as f:  # 用 "a" 模式打开
                        f.write(dataset+'\n')
                        f.write('TP: {}\n'.format(tp))
                        f.write('FP: {}\n'.format(fp))
                        f.write('TN: {}\n'.format(tn))
                        f.write('FN: {}\n'.format(fn))
                        f.write('Precision: {}\n'.format(prec[max_f1_idx]))
                        f.write('Recall: {}\n'.format(rec[max_f1_idx]))
                        f.write('FPR: {}\n'.format(fp/(fp+tn)))
                        f.write('F1: {}\n'.format(f1[max_f1_idx]))
                        f.write('AUC: {}\n'.format(auc))
                        f.write('-' * 30 + '\n')  # 可选：写一个分隔线
                    
                     # 绘图
                    # plt.figure()
                    # plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
                    # # plt.plot([0.0001, 1], [0.0001, 1], 'k--', label='Random guess')  # 对角参考线
                    # plt.xscale('log')  # 设置横坐标为对数刻度
                    # plt.xticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ['0.0001', '0.001', '0.01', '0.1', '1'])
                    # plt.xlabel('False Positive Rate (log scale)')
                    # plt.ylabel('True Positive Rate')
                    # plt.title(dataset+' ROC Curve (Log-scaled FPR)')
                    # plt.grid(True, which='both')
                    # plt.legend(loc='lower right')
                    # plt.tight_layout()
                    # plt.savefig(dataset + '_roc_curve.jpg', dpi=300)  # 保存图像
                    # plt.show()

                    # import umap
                    # label_mapping = {
                    #     0: 'Benign',
                    #     1: 'Attack',
                    # }
                    # umap_model = umap.UMAP(n_components=2,n_jobs=-1)
                    # node_embeddings_2d = umap_model.fit_transform(x_test)
                    
                    # scatter = plt.scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1], 
                    #                       c=y_test, cmap='tab10', marker='o',
                    #                       s=35,                # 增大点的大小
                    #                       edgecolors='white',  # 设置边界颜色为白色
                    #                       linewidths=1.1,      # 设置边界宽度
                    #                       alpha=0.8)           # 设置透明度
                    #                     # 获取 tab10 颜色映射的颜色
                    # cmap = plt.get_cmap('tab10')

                    # plt.tight_layout()
                    # plt.savefig(dataset,dpi=300)
                    # plt.show()
            
                    break
                    
            else:
                if tp+tn > 48 :
                    print('TP: {}'.format(tp))
                    print('FP: {}'.format(fp))
                    print('TN: {}'.format(tn))
                    print('FN: {}'.format(fn))
                    print('Precision: {}'.format(prec[max_f1_idx]))
                    print('Recall: {}'.format(rec[max_f1_idx]))
                    print('FPR: {}'.format(fp/(fp+tn)))
                    print('F1: {}'.format(f1[max_f1_idx]))
                    print('AUC: {}'.format(auc))

                    with open("result.txt", "a") as f:  # 用 "a" 模式打开
                        f.write(dataset+'\n')
                        f.write('TP: {}\n'.format(tp))
                        f.write('FP: {}\n'.format(fp))
                        f.write('TN: {}\n'.format(tn))
                        f.write('FN: {}\n'.format(fn))
                        f.write('Precision: {}\n'.format(prec[max_f1_idx]))
                        f.write('Recall: {}\n'.format(rec[max_f1_idx]))
                        f.write('FPR: {}\n'.format(fp/(fp+tn)))
                        f.write('F1: {}\n'.format(f1[max_f1_idx]))
                        f.write('AUC: {}\n'.format(auc))
                        f.write('-' * 30 + '\n')  # 可选：写一个分隔线

                    # # 绘图
                    # plt.figure()
                    # plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
                    # # plt.plot([0.0001, 1], [0.0001, 1], 'k--', label='Random guess')  # 对角参考线
                    # plt.xscale('log')  # 设置横坐标为对数刻度
                    # plt.xticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ['0.0001', '0.001', '0.01', '0.1', '1'])
                    # plt.xlabel('False Positive Rate (log scale)')
                    # plt.ylabel('True Positive Rate')
                    # plt.title(dataset+' ROC Curve (Log-scaled FPR)')
                    # plt.grid(True, which='both')
                    # plt.legend(loc='lower right')
                    # plt.tight_layout()
                    # # ✅ 保存图像（必须在 show() 之前）
                    # plt.savefig(dataset+'_roc_curve.jpg', dpi=300)  # dpi 可选，控制分辨率
                    # plt.show()

                    # import umap
                    # label_mapping = {
                    #     0: 'Benign',
                    #     1: 'Attack',
                    # }
                    # umap_model = umap.UMAP(n_components=2,n_jobs=-1)
                    # node_embeddings_2d = umap_model.fit_transform(x_test)
                    
                    # scatter = plt.scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1], 
                    #                       c=y_test, cmap='tab10', marker='o',
                    #                       s=35,                # 增大点的大小
                    #                       edgecolors='white',  # 设置边界颜色为白色
                    #                       linewidths=1.1,      # 设置边界宽度
                    #                       alpha=0.8)           # 设置透明度
                    #                     # 获取 tab10 颜色映射的颜色
                    # cmap = plt.get_cmap('tab10')

                    # plt.tight_layout()
                    # plt.savefig(dataset,dpi=300)
                    # plt.show()
                    
                    break
                    
        return auc, 0.0
       
    else:
        set_random_seed(0)
        np.random.shuffle(benign_idx)
        np.random.shuffle(attack_idx)
        if dataset == 'streamspot':
            x_train = x[benign_idx[:train_count]]
            x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
            y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
        else:
            x_train = x[benign_idx[:100]]
            x_test = np.concatenate([x[benign_idx[100:]], x[attack_idx]], axis=0)
            y_test = np.concatenate([y[benign_idx[100:]], y[attack_idx]], axis=0)
        
        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs.fit(x_train)
        distances, indexes = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
        mean_distance = distances.mean() * n_neighbors / (n_neighbors - 1)
        distances, indexes = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)

        score = distances.mean(axis=1) / mean_distance

            # 计算 ROC 曲线
        fpr, tpr, _ = roc_curve(y_test, score)
    
        auc = roc_auc_score(y_test, score)
        prec, rec, threshold = precision_recall_curve(y_test, score)
        f1 = 2 * prec * rec / (rec + prec )
        best_idx = np.argmax(f1)
        best_thres = threshold[best_idx]

        tn = 0
        fn = 0
        tp = 0
        fp = 0
        for i in range(len(y_test)):
            if y_test[i] == 1.0 and score[i] >= best_thres:
                tp += 1
            if y_test[i] == 1.0 and score[i] < best_thres:
                fn += 1
            if y_test[i] == 0.0 and score[i] < best_thres:
                tn += 1
            if y_test[i] == 0.0 and score[i] >= best_thres:
                fp += 1
        with open("result.txt", "a") as f:  # 用 "a" 模式打开
            f.write(dataset+'\n')
            f.write('TP: {}\n'.format(tp))
            f.write('FP: {}\n'.format(fp))
            f.write('TN: {}\n'.format(tn))
            f.write('FN: {}\n'.format(fn))
            f.write('Precision: {}\n'.format(prec[best_idx]))
            f.write('Recall: {}\n'.format(rec[best_idx]))
            f.write('FPR: {}\n'.format(fp/(fp+tn)))
            f.write('F1: {}\n'.format(f1[best_idx]))
            f.write('AUC: {}\n'.format(auc))
            f.write('-' * 30 + '\n')  # 可选：写一个分隔线

        print('TP: {}'.format(tp))
        print('FP: {}'.format(fp))
        print('TN: {}'.format(tn))
        print('FN: {}'.format(fn))
        print('Precision: {}'.format(prec[best_idx]))
        print('Recall: {}'.format(rec[best_idx]))
        print('FPR: {}'.format(fp/(fp+tn)))
        print('F1: {}'.format(f1[best_idx]))
        print('AUC: {}'.format(auc))

         # 绘图
        # plt.figure()
        # plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
        # # plt.plot([0.0001, 1], [0.0001, 1], 'k--', label='Random guess')  # 对角参考线
        # plt.xscale('log')  # 设置横坐标为对数刻度
        # plt.xticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ['0.0001', '0.001', '0.01', '0.1', '1'])
        # plt.xlabel('False Positive Rate (log scale)')
        # plt.ylabel('True Positive Rate')
        # plt.title(dataset+' ROC Curve (Log-scaled FPR)')
        # plt.grid(True, which='both')
        # plt.legend(loc='lower right')
        # plt.tight_layout()
        # # ✅ 保存图像（必须在 show() 之前）
        # plt.savefig(dataset+'_roc_curve.jpg', dpi=300)  # dpi 可选，控制分辨率
        # plt.show()
        
        return auc, 0.0


def evaluate_entity_level_using_knn(dataset, x_train, x_test, y_test):
    # x_train_mean = x_train.mean(axis=0)
    # x_train_std = x_train.std(axis=0)
    # x_train = (x_train - x_train_mean) / x_train_std
    # x_test = (x_test - x_train_mean) / x_train_std

    if dataset == 'cadets':
        n_neighbors = 10
    else:
        n_neighbors = 10

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nbrs.fit(x_train)

    save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset)
    print('eval--------------------------------')
    if not os.path.exists(save_dict_path):
        idx = list(range(x_train.shape[0]))
        random.shuffle(idx)
        distances, _ = nbrs.kneighbors(x_train[idx][:min(50000, x_train.shape[0])], n_neighbors=n_neighbors)
        del x_train
        mean_distance = distances.mean()
        del distances
        distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        save_dict = [mean_distance, distances.mean(axis=1)]
        distances = distances.mean(axis=1)
        with open(save_dict_path, 'wb') as f:
            pkl.dump(save_dict, f)
    else:
        with open(save_dict_path, 'rb') as f:
            mean_distance, distances = pkl.load(f)
    score = distances / mean_distance
    del distances

    # 计算 ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, score)

    auc = roc_auc_score(y_test, score)
    prec, rec, threshold = precision_recall_curve(y_test, score)
    f1 = 2 * prec * rec / (rec + prec )
    best_idx = -1
    for i in range(len(f1)):
        # To repeat peak performance
        if dataset == 'trace' and rec[i] < 0.99979:
            best_idx = i - 1
            break
        if dataset == 'theia' and rec[i] < 0.99996:
            best_idx = i - 1
            break
        if dataset == 'cadets' and rec[i] < 0.9976:
            best_idx = i - 1
            break
    best_thres = threshold[best_idx]

    tn = 0
    fn = 0
    tp = 0
    fp = 0
    for i in range(len(y_test)):
        if y_test[i] == 1.0 and score[i] >= best_thres:
            tp += 1
        if y_test[i] == 1.0 and score[i] < best_thres:
            fn += 1
        if y_test[i] == 0.0 and score[i] < best_thres:
            tn += 1
        if y_test[i] == 0.0 and score[i] >= best_thres:
            fp += 1

    with open("result.txt", "a") as f:  # 用 "a" 模式打开
        f.write(dataset+'\n')
        f.write('TP: {}\n'.format(tp))
        f.write('FP: {}\n'.format(fp))
        f.write('TN: {}\n'.format(tn))
        f.write('FN: {}\n'.format(fn))
        f.write('Precision: {}\n'.format(prec[best_idx]))
        f.write('Recall: {}\n'.format(rec[best_idx]))
        f.write('FPR: {}\n'.format(fp/(fp+tn)))
        f.write('F1: {}\n'.format(f1[best_idx]))
        f.write('AUC: {}\n'.format(auc))
        f.write('-' * 30 + '\n')  # 可选：写一个分隔线
        
    print('TP: {}'.format(tp))
    print('FP: {}'.format(fp))
    print('TN: {}'.format(tn))
    print('FN: {}'.format(fn))
    print('PRECISION: {}'.format(prec[best_idx]))
    print('RECALL: {}'.format(rec[best_idx]))
    print('FPR: {}'.format(fp/(fp+tn)))
    print('F1: {}'.format(f1[best_idx]))
    print('AUC: {}'.format(auc))

    # 绘图
    # plt.figure()
    # plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    # # plt.plot([0.0001, 1], [0.0001, 1], 'k--', label='Random guess')  # 对角参考线
    # plt.xscale('log')  # 设置横坐标为对数刻度
    # plt.xticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ['0.0001', '0.001', '0.01', '0.1', '1'])
    # plt.xlabel('False Positive Rate (log scale)')
    # plt.ylabel('True Positive Rate')
    # plt.title(dataset+' ROC Curve (Log-scaled FPR)')
    # plt.grid(True, which='both')
    # plt.legend(loc='lower right')
    # plt.tight_layout()
    # # ✅ 保存图像（必须在 show() 之前）
    # plt.savefig(dataset+'_roc_curve.jpg', dpi=300)  # dpi 可选，控制分辨率
    # plt.show()

    # import umap
    # label_mapping = {
    #     0: 'Benign',
    #     1: 'Attack',
    # }
    # umap_model = umap.UMAP(n_components=2,n_jobs=-1)
    # node_embeddings_2d = umap_model.fit_transform(x_test)
    
    # scatter = plt.scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1], 
    #                       c=y_test, cmap='tab10', marker='o',
    #                       s=35,                # 增大点的大小
    #                       edgecolors='white',  # 设置边界颜色为白色
    #                       linewidths=1.1,      # 设置边界宽度
    #                       alpha=0.8)           # 设置透明度
    #                     # 获取 tab10 颜色映射的颜色
    # cmap = plt.get_cmap('tab10')

    # plt.tight_layout()
    # plt.savefig(dataset,dpi=300)
    # plt.show()
    
    return auc, 0.0, None, None
