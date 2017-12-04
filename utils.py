from __future__ import print_function

import scipy.sparse as sp
import numpy as np


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(map(classes_dict.get, labels), dtype=np.int32)
    return labels_onehot


def load_data(path="data/rita/", dataset="rita",
                           normalizer=None,
                           max_adjacency=float("inf"),
                           symmetric=True,
                           add_node_one_hot=False,
                           self_links=1):
    """Load graph dataset"""
    print('Loading {} dataset...'.format(dataset))

    # for aifb and mutag the train/test splits are predefined
    if dataset in ['aifb', 'mutag', 'rita_tts', 'rita_tts_hard', 'rita_tts_hard_lstm', 'rita_tts_lstm', 'nell_tts']:
        idx_features_labels = np.loadtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-2], dtype=np.float32).todense()
        labels = encode_onehot(idx_features_labels[:, -2])
        train_test_idx = idx_features_labels[:,-1]
    else:
        idx_features_labels = np.loadtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32).todense()
        labels = encode_onehot(idx_features_labels[:, -1])
        train_test_idx = None #these will be determined later during get_splits()

    # the old and dense way of adding a one hot vector as node features
    # if add_node_one_hot == True:
    #     features = np.hstack([features, np.eye(features.shape[0])])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_loaded = np.loadtxt("{}{}.cites".format(path, dataset), dtype=np.float32)

    # process different adjacency matrices

    # calculate total amount of adjacency matrices (undirected)
    adjs = []
    nb_adjs = max(1, edges_loaded.shape[1] - 2)

    # extract the edges
    edges_unordered = edges_loaded[:,0:2]
    edges = np.array(map(idx_map.get, edges_unordered[:,0:2].flatten()),
    dtype=np.int32).reshape(edges_unordered.shape)

    # normalize the data columnwise
    edge_features = edges_loaded
    edge_features[:,0:2] = edges

    # normalize
    if normalizer is not None:
        if features.shape[1] > 0:
            features = normalizer.fit_transform(features)
        if edge_features[:,2:].shape[1] > 0:
            edge_features[:,2:] = normalizer.fit_transform(edge_features[:,2:])


    # process the different adjacency matrices
    for a in range(nb_adjs):
        #extract the weights for the links in this adjacency matrix
        if edges_loaded.shape[1] > 2 and max_adjacency > 0:
            #if weights are specified
            edges_weights = edge_features[:,a+2]
            #if all edge weights are 0 we skip this one
            ew_abs_sum = np.sum(np.abs(edges_weights))
            if ew_abs_sum == 0.0:
                continue
        else:
            #if no weights are specified
            edges_weights = np.ones([edges_loaded.shape[0]])

        if symmetric == True:
            adj = sp.coo_matrix(   (
                                    np.append(edges_weights, edges_weights),
                                    (
                                        np.append(edges[:, 0], edges[:, 1]),
                                        np.append(edges[:, 1], edges[:, 0])
                                    )
                                ),
                                shape=(labels.shape[0], labels.shape[0]),
                                dtype=np.float32
                            )
            #if there are no separate self links, we have to add them here
            if self_links == 0:
                adj = adj + sp.eye(adj.shape[0])*2

            # build symmetric adjacency matrix
            adj.sum_duplicates()
            #add the adjacency matrix to the list
            adjs += [adj]
        else:
            adjs += [sp.csr_matrix(   (
                                    edges_weights,
                                    (
                                        edges[:, 0],
                                        edges[:, 1]
                                    )
                                ),
                                shape=(labels.shape[0], labels.shape[0]),
                                dtype=np.float32
                            )]
            adjs += [sp.csr_matrix(   (
                                    edges_weights,
                                    (
                                        edges[:, 1],
                                        edges[:, 0]
                                    )
                                ),
                                shape=(labels.shape[0], labels.shape[0]),
                                dtype=np.float32
                            )]


        if a + 1 >= max_adjacency:
            print(len(adjs))
            break
    print('total adjacency matrices', len(adjs), 'out of', nb_adjs)

    return features, adjs, labels, train_test_idx

def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten())
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten())
        a_norm = d.dot(adj).tocsr()
    return a_norm

def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])*2
    adj = normalize_adj(adj, symmetric)
    return adj

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def get_splits_predefined(y, train_test_idx, train_split, val_split, testing=False):
    train_test_idx = np.array([int(x) for x in train_test_idx])
    predef_idx_train = np.where(train_test_idx == 1)[0]
    if testing == True:
        predef_idx_test  = np.where(train_test_idx == 2)[0]

    total       = predef_idx_train.shape[0]
    train_split = train_split / 100.0
    val_split   = val_split / 100.0
    train_split = int(total * train_split)
    val_split   = train_split + int(total * val_split)

    indices = range(0, total)

    np.random.shuffle(indices)

    if testing == False:
        idx_train   = predef_idx_train[indices[0:train_split]]
        idx_val     = predef_idx_train[indices[train_split:val_split]]
        idx_test    = predef_idx_train[indices[val_split:int(total)]]
    elif testing == True:
        idx_train   = predef_idx_train[indices[0:train_split]]
        idx_val     = predef_idx_train[indices[train_split:]]
        idx_test    = predef_idx_test

    y_train     = np.zeros(y.shape, dtype=np.int32)
    y_val       = np.zeros(y.shape, dtype=np.int32)
    y_test      = np.zeros(y.shape, dtype=np.int32)

    y_train[idx_train]  = y[idx_train]
    y_val[idx_val]      = y[idx_val]
    y_test[idx_test]    = y[idx_test]

    train_mask = sample_mask(idx_train, y.shape[0])
    print('nodes in train set:  ', idx_train.shape, '\tclass distribution: ', np.sum(y_train, axis=0))
    print('nodes in val set:    ', idx_val.shape, '\tclass distribution: ', np.sum(y_val, axis=0))
    print('nodes in test set:   ', idx_test.shape, '\tclass distribution: ', np.sum(y_test, axis=0))
    print('nodes in training mask, should be the same as train set:', sum(train_mask))

    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask

def get_splits(y):
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask

def get_splits_weighted(y, train_split, val_split):
    total       = y.shape[0]
    train_split = train_split / 100.0
    val_split   = val_split / 100.0
    train_split = int(total * train_split)
    val_split   = train_split + int(total * val_split)

    indices = range(0, total)
    np.random.shuffle(indices)

    idx_train   = indices[0:train_split]
    idx_val     = indices[train_split:val_split]
    idx_test    = indices[val_split:int(total)]

    y_train     = np.zeros(y.shape, dtype=np.int32)
    y_val       = np.zeros(y.shape, dtype=np.int32)
    y_test      = np.zeros(y.shape, dtype=np.int32)

    y_train[idx_train]  = y[idx_train]
    y_val[idx_val]      = y[idx_val]
    y_test[idx_test]    = y[idx_test]

    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def categorical_crossentropy(preds, labels):
    epsilon = 1e-8
    return np.mean(-np.log(np.maximum(np.extract(labels, preds), epsilon)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
