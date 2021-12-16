import os
import copy
import numpy as np
import cv2
import torch
from tqdm import tqdm
from . import cdml
import tensorflow as tf
import tensorboard as tb

import numpy as np
import math
from scipy.special import comb

from sklearn import cluster
from sklearn import metrics
from sklearn import neighbors


def evaluate_triplet(data_streams, writer, max_steps, n_class,
                     pretrained=False,
                     model_path='model',
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                     epoch=30000):
    stream_train, stream_train_eval, stream_test = data_streams
    epoch_iterator = stream_test.get_epoch_iterator()
    test_data = []
    test_label = []
    test_img = []

    tri = cdml.TripletBase(n_class=n_class, pretrained=pretrained).to(device)
    checkpoint = torch.load(os.path.join(model_path, 'model_%d.pth' % epoch))
    tri.load_state_dict(checkpoint)
    tri.eval()

    img_mean = np.array([123, 117, 104], dtype=np.float32).reshape(1, 3, 1, 1)
    cnt = 0

    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

    with tqdm(total=max_steps) as pbar:
        for batch in copy.copy(epoch_iterator):
            x_batch, label = batch
            x_batch -= img_mean
            pbar.update(1)
            embedding_z = tri(torch.from_numpy(x_batch).to(device), False)
            test_data.extend(embedding_z.detach().cpu().numpy())
            test_label.extend(label)
            for x in x_batch:
                xx = cv2.resize(x.transpose(1, 2, 0), (32, 32)).transpose(2, 0, 1)
                test_img.append((xx + img_mean[0]) / 255.0)
            cnt += 1
            if cnt > max_steps:
                break

        nmi, f1 = evaluate_cluster(test_data, test_label, 98)
        print(nmi, f1)

        writer.add_embedding(np.vstack(test_data), np.vstack(test_label).flatten(),
                             torch.from_numpy(np.stack(test_img, axis=0)),
                             global_step=cnt, tag='embedding/test')
        writer.flush()

def evaluate_fc(data_streams, writer, max_steps, n_class,
                     pretrained=False,
                     model_path='model',
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                     epoch=30000):
    stream_train, stream_train_eval, stream_test = data_streams
    epoch_iterator = stream_test.get_epoch_iterator()
    test_data = []
    test_label = []
    test_img = []

    tri = cdml.TripletFC(n_class=n_class, pretrained=pretrained).to(device)
    checkpoint = torch.load(os.path.join(model_path, 'model_fc_%d.pth' % epoch))
    tri.load_state_dict(checkpoint)
    tri.eval()

    img_mean = np.array([123, 117, 104], dtype=np.float32).reshape(1, 3, 1, 1)
    cnt = 0

    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

    with tqdm(total=max_steps) as pbar:
        for batch in copy.copy(epoch_iterator):
            x_batch, label = batch
            x_batch -= img_mean
            pbar.update(1)
            embedding_z = tri.embedding(torch.from_numpy(x_batch).to(device), False)
            test_data.extend(embedding_z.detach().cpu().numpy())
            test_label.extend(label)
            for x in x_batch:
                xx = cv2.resize(x.transpose(1, 2, 0), (32, 32)).transpose(2, 0, 1)
                test_img.append((xx + img_mean[0]) / 255.0)
            cnt += 1
            if cnt > max_steps:
                break

        neighbors = [1, 2, 4, 8]
        nmi, f1 = evaluate_cluster(test_data, test_label, n_class - 1)
        recall = evaluate_recall(np.array(test_data), test_label,np.array(neighbors))
        print("nmi", nmi, "f1", f1)

        writer.add_embedding(np.vstack(test_data), np.vstack(test_label).flatten(),
                             torch.from_numpy(np.stack(test_img, axis=0)),
                             global_step=cnt, tag='embedding/test')
        writer.flush()


def evaluate_cdml_triplet(data_streams, writer, max_steps, n_class,
                          pretrained=False,
                          loss='triplet',
                          model_path='model',
                          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                          epoch=30000, n_cand=4, alpha=0.1, beta=1):
    stream_train, stream_train_eval, stream_test = data_streams
    epoch_iterator = stream_test.get_epoch_iterator()
    test_data = []
    test_label = []
    test_img = []

    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

    checkpoint = torch.load(os.path.join(model_path, 'model_alpha%.2f_cand%d_epoch%d.pth' % (alpha, n_cand, epoch)))

    cdml_net = cdml.TripletCDML(n_class=n_class, pretrained=pretrained, alpha=alpha, beta=beta).to(device)

    cdml_net.load_state_dict(checkpoint['model_state_dict'])
    cdml_net.eval()

    img_mean = np.array([123, 117, 104], dtype=np.float32).reshape(1, 3, 1, 1)
    cnt = 0


    with tqdm(total=max_steps) as pbar:
        for batch in copy.copy(epoch_iterator):
            x_batch, label = batch
            x_batch -= img_mean
            pbar.update(1)
            embedding_z = cdml_net.embedding(torch.from_numpy(x_batch).to(device), False)
            test_data.extend(embedding_z.detach().cpu().numpy())
            test_label.extend(label)
            for x in x_batch:
                xx = cv2.resize(x.transpose(1, 2, 0), (32, 32)).transpose(2, 0, 1)
                test_img.append((xx + img_mean[0]) / 255.0)
            cnt += 1
            if cnt > max_steps:
                break

        neighbors=[1,2,4,8]
        nmi, f1 = evaluate_cluster(test_data, test_label, n_class-1)
        recall = evaluate_recall(np.array(test_data), test_label,np.array(neighbors))
        print("nmi",nmi,"f1", f1)


        writer.add_embedding(np.vstack(test_data), np.vstack(test_label).flatten(),
                             torch.from_numpy(np.stack(test_img, axis=0)),
                             global_step=cnt, tag='embedding/test')
        writer.flush()

# return nmi,f1; n_cluster = num of classes
def evaluate_cluster(feats,labels,n_clusters):

    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0).fit(feats)
    centers = kmeans.cluster_centers_

    ### k-nearest neighbors
    neigh = neighbors.KNeighborsClassifier(n_neighbors=1)
    neigh.fit(centers,range(len(centers)))

    idx_in_centers = neigh.predict(feats)
    num = len(feats)
    d = np.zeros(num)
    for i in range(num):
        d[i] = np.linalg.norm(feats[i] - centers[idx_in_centers[i],:])

    labels_pred = np.zeros(num)
    for i in range(n_clusters):
        index = np.where(idx_in_centers == i)[0];
        ind = np.argmin(d[index]);
        cid = index[ind];
        labels_pred[index] = cid;


    nmi,f1 =  compute_clutering_metric(labels, labels_pred)
    return nmi,f1


def compute_clutering_metric(idx, item_ids):
    N = len(idx);

    centers = np.unique(idx);
    num_cluster = len(centers);

    # count the number of objects in each cluster
    count_cluster = np.zeros((num_cluster));
    for i in range(num_cluster):
        count_cluster[i] = len(np.where(idx == centers[i])[0]);

    # build a mapping from item_id to item index
    keys = np.unique(item_ids);
    num_item = len(keys);
    values = range(num_item);
    item_map = dict();
    for i in range(len(keys)):
        item_map.update([(keys[i], values[i])])

    # count the number of objects of each item
    count_item = np.zeros(num_item);
    for i in range(N):
        index = item_map[item_ids[i]];
        count_item[index] = count_item[index] + 1;

    # compute purity
    purity = 0;
    for i in range(num_cluster):
        member = np.where(idx == centers[i])[0];
        member_ids = item_ids[member];

        count = np.zeros(num_item);
        for j in range(len(member)):
            index = item_map[member_ids[j]];
            count[index] = count[index] + 1;
        purity = purity + max(count);

    purity = purity / N;

    # compute Normalized Mutual Information (NMI)
    count_cross = np.zeros((num_cluster, num_item));
    for i in range(N):
        index_cluster = np.where(idx[i] == centers)[0];
        index_item = item_map[item_ids[i]];
        count_cross[index_cluster, index_item] = count_cross[index_cluster, index_item] + 1;

    # mutual information
    I = 0;
    for k in range(num_cluster):
        for j in range(num_item):
            if count_cross[k, j] > 0:
                s = count_cross[k, j] / N * math.log(N * count_cross[k, j] / (count_cluster[k] * count_item[j]));
                I = I + s;

    # entropy
    H_cluster = 0;
    for k in range(num_cluster):
        s = -count_cluster[k] / N * math.log(count_cluster[k] / float(N));
        H_cluster = H_cluster + s;

    H_item = 0;
    for j in range(num_item):
        s = -count_item[j] / N * math.log(count_item[j] / float(N));
        H_item = H_item + s;

    NMI = 2 * I / (H_cluster + H_item);

    # compute True Positive (TP) plus False Positive (FP)
    tp_fp = 0;
    for k in range(num_cluster):
        if count_cluster[k] > 1:
            tp_fp = tp_fp + comb(count_cluster[k], 2);

    # compute True Positive (TP)
    tp = 0;
    for k in range(num_cluster):
        member = np.where(idx == centers[k])[0];
        member_ids = item_ids[member];

        count = np.zeros(num_item);
        for j in range(len(member)):
            index = item_map[member_ids[j]];
            count[index] = count[index] + 1;

        for i in range(num_item):
            if count[i] > 1:
                tp = tp + comb(count[i], 2);

    # False Positive (FP)
    fp = tp_fp - tp;

    # compute False Negative (FN)
    count = 0;
    for j in range(num_item):
        if count_item[j] > 1:
            count = count + comb(count_item[j], 2);

    fn = count - tp;

    # compute True Negative (TN)
    tn = N * (N - 1) / 2 - tp - fp - fn;

    # compute RI
    RI = (tp + tn) / (tp + fp + fn + tn);

    # compute F measure
    P = tp / (tp + fp);
    R = tp / (tp + fn);
    beta = 1;
    F = (beta * beta + 1) * P * R / (beta * beta * P + R);

    return NMI, F

def distance_matrix(X):
    X = np.matrix(X)
    m = X.shape[0]
    t = np.matrix(np.ones([m, 1]))
    x = np.matrix(np.empty([m, 1]))
    for i in range(0, m):
        n = np.linalg.norm(X[i, :])
        x[i] = n * n
    D = x * np.transpose(t) + t * np.transpose(x) - 2 * X * np.transpose(X)
    return D

def evaluate_recall(features, labels, neighbours):
    """
    A function that calculate the recall score of a embedding
    :param features: The 2-d array of the embedding
    :param labels: The 1-d array of the label
    :param neighbours: A 1-d array contains X in Recall@X
    :return: A 1-d array of the Recall@X
    """
    dims = features.shape
    recalls = []
    D2 = distance_matrix(features)

    # set diagonal to very high number
    num = dims[0]
    D = np.sqrt(np.abs(D2))
    diagn = np.diag([float('inf') for i in range(0, D.shape[0])])
    D = D + diagn
    for i in range(0, np.shape(neighbours)[0]):
        recall_i = compute_recall_at_K(D, neighbours[i], labels, num)
        recalls.append(recall_i)
    print('done')
    return recalls

def compute_recall_at_K(D, K, class_ids, num):
    num_correct = 0
    for i in range(0, num):
        this_gt_class_idx = class_ids[i]
        this_row = D[i, :]
        inds = np.array(np.argsort(this_row))[0]
        knn_inds = inds[0:K]
        knn_class_inds = [class_ids[i] for i in knn_inds]
        if sum(np.in1d(knn_class_inds, this_gt_class_idx)) > 0:
            num_correct = num_correct + 1
    recall = float(num_correct)/float(num)

    print("K: %d, Recall: %.3f\n" % (K, recall))
    return recall