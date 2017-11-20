from __future__ import print_function

# import xrange
from keras.layers import Input, Dropout, Dense, Activation, LSTM, SimpleRNN, GRU, Conv1D, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.constraints import non_neg
from keras import initializers
from keras.regularizers import l2
from keras.layers.core import Lambda
from layers.graph import GraphConvolution, GraphInput, get_tensor_shape, \
    vector_to_adjacency, vector_to_adjacency_normalized, vector_to_adjacency_sym_normalized, \
    extract_from_adjs, reshape_for_lstm, vector_to_adjacency_sym_sparse, vector_to_adjacency_softmax
from utils import *

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

import numpy as np
import argparse
import time


def experiment(args):
    # Get data
    normalizer = preprocessing.StandardScaler()

    if args.FILTER in ['localpool', 'dense']:
        X, A, y, train_test_idx = load_data(path=args.PATH,
                                                dataset=args.DATASET,
                                                normalizer=normalizer,
                                                max_adjacency=0,
                                                symmetric=args.SYMMETRIC,
                                                add_node_one_hot=args.ADD_NODE_ONE_HOT)

    elif args.FILTER in ['efgcn', 'lgcn']:
        X, A, y, train_test_idx = load_data(path=args.PATH,
                                                dataset=args.DATASET,
                                                normalizer=normalizer,
                                                max_adjacency=args.MAX_ADJACENCY,
                                                symmetric=args.SYMMETRIC,
                                                add_node_one_hot=args.ADD_NODE_ONE_HOT,
                                                self_links=args.SELF_LINKS)

    else:
        raise Exception('Invalid filter type for loading data')

    if args.DATASET in ['aifb', 'mutag', 'rita_tts', 'rita_tts_hard', 'rita_tts_hard_lstm', 'rita_tts_lstm', 'nell_tts']:
        y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits_predefined(y,train_test_idx,
                                                                                                args.TRAIN_SPLIT,
                                                                                                args.VAL_SPLIT,
                                                                                                args.TESTING)
    elif args.DATASET in ['cora', 'cora_plus']:
        y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)
    else:
        y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits_weighted(y, args.TRAIN_SPLIT,
                                                                                                args.VAL_SPLIT)
    batch_size=X.shape[0] # number of nodes

    if args.FILTER == 'localpool':
        """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
        if args.VERBOSE >= 1: print('Using local pooling filters...')
        A_ = preprocess_adj(A[0], args.SYM_NORM)
        support = 1
        graph = [X, A_]
        G = [GraphInput(sparse=True, name='adjacency')]

    elif args.FILTER == 'dense':
        """Running a regular dense network """
        if args.VERBOSE >= 1: print('Running a regular dense network')
        graph = [X]
        G = []

    elif args.FILTER in ['efgcn','lgcn']:
        """Running the Latent-Graph Convolutional Network algorithm """
        if args.VERBOSE >= 1: print('Splitting up adjacency matrices...')
        support = len(A)
        A = [a.tocsr() for a in A]
        graph = [X]+A
        G = [GraphInput(sparse=True, name='adjacency_'+str(s)) for s in range(support)]

    else:
        raise Exception('Invalid filter type for creating processing data')


    # Define model architecture

    X_in = Input(shape=(X.shape[1],))
    H = Dropout(args.DROPOUT)(X_in)

    if args.FILTER in ['localpool']:
        for hidden_nodes in args.NETWORK_LAYERS:
            H = GraphConvolution(hidden_nodes,
                                    support,
                                    activation=args.ACTIVATION,
                                    kernel_regularizer=l2(args.REG_STRENGTH),
                                    use_bias=True,
                                    self_links=args.SELF_LINKS)([H]+G)
            H = Dropout(args.DROPOUT)(H)
        Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)

    elif args.FILTER in ['efgcn', 'lgcn']:

        # selecting the normalize function
        if args.ADJ_NORMALIZER == 'sym':
            vector_to_adjacency_function = vector_to_adjacency_sym_normalized
        elif args.ADJ_NORMALIZER == 'right':
            vector_to_adjacency_function = vector_to_adjacency_normalized
        elif args.ADJ_NORMALIZER == 'none':
            vector_to_adjacency_function = vector_to_adjacency
        elif args.ADJ_NORMALIZER == 'sym_sparse':
            vector_to_adjacency_function = vector_to_adjacency_sym_sparse
        elif args.ADJ_NORMALIZER == 'softmax':
            vector_to_adjacency_function = vector_to_adjacency_softmax
        else:
            raise Exception('Invalid normalizing mode for L-GCN')

        # turn sparse adjacency tensor into dense edge features matrix
        def latent_relation_layer(A, G, args, name):
            He = Lambda(extract_from_adjs, output_shape=(A[0].data.shape[0], len(G)))(G)


            if args.FILTER == 'lgcn':
                # embedding hidden layers
                for i, embedding_len in enumerate(args.EMBEDDING_LAYERS):
                    # add dropout if specified
                    if args.EMB_DROPOUT > 0.0:
                        He = Dropout(args.EMB_DROPOUT)(He)

                    # if it is the rita lstm dataset 12 months 20 edge features per month
                    if args.DATASET in ['rita_lstm', 'rita_tts_lstm', 'rita_tts_hard_lstm']:
                        output_shape = (12, 20)
                        He = Lambda(reshape_for_lstm)(He)
                        He = LSTM(units=embedding_len,
                                    activation=args.EMBEDDING_ACT,
                                    input_shape=output_shape,
                                    kernel_initializer=initializers.RandomUniform())(He)
                    else:
                        He = Dense(embedding_len,
                                   activation=args.EMBEDDING_ACT,
                                #    kernel_initializer=initializers.RandomUniform(),
                                   name='latent_relation_' + name + '_{}'.format(i))(He)



            elif args.FILTER == 'efgcn':
                embedding_len = len(G)
                He = Activation('relu')(He)

            # helper functions for the vector_to_adjacency_function
            # may be nicer to put these all into one Keras Layer instance, but it's not necessary
            tensor_shape = Lambda(get_tensor_shape, output_shape=(2,))(G[0])
            output_shape = (A[0].shape[0], A[0].shape[1])
            Ge = []

            # slice the dense edge feature matrix and make adjacency matrices from them
            for slice_index in xrange(embedding_len):
                #slice
                sli = Lambda(lambda x: x[:, slice_index])(He)
                #to adjacency matrices
                Ge += [Lambda(vector_to_adjacency_function, output_shape=output_shape)([G[0], sli, tensor_shape])]
            return Ge, embedding_len

        # Ge, embedding_len = latent_relation_layer(A, G, args, 'hidden')

        # loop over hidden layers args.NETWORK_LAYERS = [16]: it goes from input to 16 to output input->32->16->output = [32,16]
        for l, hidden_nodes in enumerate(args.NETWORK_LAYERS):
            # if it is the first layer, and its one_hot node features, we remove the self links
            if args.ADD_NODE_ONE_HOT == True and l == 0:
                first_layer_one_hot = True
            else:
                first_layer_one_hot = False
            Ge, embedding_len = latent_relation_layer(A, G, args, 'hidden_' + str(l))

            H = GraphConvolution(hidden_nodes,
                                 embedding_len,
                                 activation=args.ACTIVATION,
                                 kernel_regularizer=l2(args.REG_STRENGTH),
                                 use_bias=True,
                                 self_links=args.SELF_LINKS,
                                 first_layer_one_hot=first_layer_one_hot)([H]+Ge)

            H = Dropout(args.DROPOUT)(H)
        args.EMBEDDING_LAYERS = [4,1]
        Ge, embedding_len = latent_relation_layer(A, G, args, 'final')
        Y = GraphConvolution(y.shape[1], embedding_len, activation='softmax', use_bias=True, self_links=args.SELF_LINKS)([H]+Ge)


    elif args.FILTER == 'dense':
        for hidden_nodes in args.NETWORK_LAYERS:
            H = Dense(hidden_nodes,
                        activation=args.ACTIVATION,
                        kernel_regularizer=l2(args.REG_STRENGTH)
                        )(H)
            H = Dropout(args.DROPOUT)(H)
        Y = Dense(y.shape[1], activation='softmax')(H)

    else:
        raise Exception('invalid filter type for network creation')

    # Compile model
    model = Model(inputs=[X_in]+G, outputs=Y)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=args.LEARNING_RATE))
    model.summary()


    wait = 0
    preds = None
    best_val_loss = 99999



    # class balance
    if args.BALANCE_CW == True:
        class_balance = np.sum(y, axis=0)
        class_weight = {}
        class_balance = 1.0/class_balance
        class_balance = class_balance/np.mean(class_balance)
        for c in range(len(class_balance)):
            class_weight[c] = class_balance[c]
        print('class weight:', class_weight)
    else:
        class_weight = None

    # Fit
    for epoch in range(1, args.NB_EPOCH+1):

        # Log wall-clock time
        t = time.time()

        # Single training iteration (we mask nodes without labels for loss calculation)
        model.fit(graph, y_train, sample_weight=train_mask,
                  batch_size=batch_size, epochs=1, shuffle=False, verbose=0, class_weight=class_weight)

        # Predict on full dataset
        preds = model.predict(graph, batch_size=batch_size)

        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                       [idx_train, idx_val])
        if args.VERBOSE == 2:
            print("Epoch: {:04d}".format(epoch),
                  "train_loss= {:.4f}".format(train_val_loss[0]),
                  "train_acc= {:.4f}".format(train_val_acc[0]),
                  "val_loss= {:.4f}".format(train_val_loss[1]),
                  "val_acc= {:.4f}".format(train_val_acc[1]),
                  "time= {:.4f}".format(time.time() - t),
                  "stopping= {}/{}".format(wait, args.PATIENCE))

        # Early stopping
        if train_val_loss[1] < best_val_loss:
            best_val_loss = train_val_loss[1]
            wait = 0
        else:
            if wait >= args.PATIENCE:
                if args.VERBOSE >= 1:
                    print('Epoch {}: early stopping'.format(epoch))
                if args.VERBOSE == 1:
                    print("train_loss= {:.4f}".format(train_val_loss[0]),
                          "train_acc= {:.4f}".format(train_val_acc[0]),
                          "val_loss= {:.4f}".format(train_val_loss[1]),
                          "val_acc= {:.4f}".format(train_val_acc[1]),
                          "time= {:.4f}".format(time.time() - t),
                          "stopping= {}/{}".format(wait, args.PATIENCE))
                break
            wait += 1

    # Testing
    if args.VERBOSE == 2:
        print('predictions')
        for i in idx_test:
            print(preds[i], np.argmax(preds[i]), np.argmax(y_test[i]))

    if args.VERBOSE == 2:
        for layer in model.layers:
            if 'latent_relation' in layer.name:
                print(layer.name)
                print(layer.get_weights())

    test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])

    if args.VERBOSE >= 1:
        print(confusion_matrix(np.argmax(preds[idx_test], axis=1), np.argmax(y_test[idx_test], axis=1)))
        print("Test set results:",
            "loss= {:.4f}".format(test_loss[0]),
            "accuracy= {:.4f}".format(test_acc[0]))

    return test_acc

def main():
    parser = argparse.ArgumentParser(description='Run L-GCN experiments')
    parser.add_argument('--dataset',            dest='DATASET',         type=str,   default='rita',         help='dataset to run experiments on (default="rita")')
    parser.add_argument('--filter',             dest='FILTER',          type=str,   default='lgcn',         help='type of filter/mode to be used in the network (localpool, lgcn, dense) (default="lgcn")')
    parser.add_argument('--max_adjacency',      dest='MAX_ADJACENCY',   type=float, default=float("inf"),   help='maximum adjacency matrices/edge features (default=float("inf") (no limitation))')
    parser.add_argument('--adj_normalizer',     dest='ADJ_NORMALIZER',  type=str,   default='sym_sparse',   help='select the adjacency normalization technique; sym for symmetrical, right for right side only, none for no normalization (default="sym")')
    parser.add_argument('--embedding_layers',   dest='EMBEDDING_LAYERS',type=str,   default='2',            help='specify the layers for the embeddings, should be hidden nodes per layer separated by a comma (default="1")')
    parser.add_argument('--embedding_act',      dest='EMBEDDING_ACT',   type=str,   default='relu',         help='specify the activation function for the embedding layer (default="sigmoid")')
    parser.add_argument('--self_links',         dest='SELF_LINKS',      type=int,   default=0,              help='add a separate adjacency matrix for self links (default=0)')
    parser.add_argument('--sym_norm',           dest='SYM_NORM',        type=int,   default=1,              help='symmetrical normalization 1 for yes, 0 for no (default=1)')
    parser.add_argument('--nb_epoch',           dest='NB_EPOCH',        type=int,   default=500,            help='number of epochs for training (default=500)')
    parser.add_argument('--patience',           dest='PATIENCE',        type=int,   default=20,             help='patience before early stopping (default=100)')
    parser.add_argument('--network_layers',     dest='NETWORK_LAYERS',  type=str,   default='16',           help='create the network structure, should be hidden nodes per layer separated by a comma (default="16") <- 1 layer 16 hidden nodes. For 2 hidden layers for example: "32,16"')
    parser.add_argument('--dropout',            dest='DROPOUT',         type=int,   default=50,             help='dropout amount (in integer percentages): 20 == Dropout(0.2) (default=50)')
    parser.add_argument('--emb_dropout',        dest='EMB_DROPOUT',     type=int,   default=0,              help='dropout amount for the embedding layers (in integer percentages): 20 == Dropout(0.2) (default=0)')
    parser.add_argument('--train_split',        dest='TRAIN_SPLIT',     type=int,   default=40,             help='train split of the data (in integer percentages) (default=20)')
    parser.add_argument('--val_split',          dest='VAL_SPLIT',       type=int,   default=10,             help='test split of the data (in integer percentages) (default=10)')
    parser.add_argument('--activation',         dest='ACTIVATION',      type=str,   default='relu',         help='specify the activation keras activation function for the hidden layers (default="relu")')
    parser.add_argument('--reg_strength',       dest='REG_STRENGTH',    type=float, default=5e-4,           help='specify the regularization strength for the l2 regularization (default=5e-4)')
    parser.add_argument('--verbose',            dest='VERBOSE',         type=int,   default=2,              help='specify the verbosity of the experiment 0 for no prints, 1 for some prints, 2 for all prints (default=0)')
    parser.add_argument('--seed',               dest='RANDOM_SEED',     type=int,   default=42,             help='specify a random seed for the train/val/test splits (default=42)')
    parser.add_argument('--runs',               dest='RUNS',            type=int,   default=0,              help='specify the number of runs with random seeds')
    parser.add_argument('--balance_cw',         dest='BALANCE_CW',      type=int,   default=0,              help='balance class weight during training time')
    parser.add_argument('--learning_rate',      dest='LEARNING_RATE',   type=float, default=0.01,           help='specify the learning rate (default=0.01)')
    parser.add_argument('--symmetric',          dest='SYMMETRIC',       type=int,   default=1,              help='specify if adjacency matrices should be made symmetrical (default=1)')
    parser.add_argument('--add_node_one_hot',   dest='ADD_NODE_ONE_HOT',type=int,   default=0,              help='add node one hot to node features (default=0)')
    # parser.add_argument('--testing',            dest='TESTING',         type=int,   default=0,              help='specify if you want to perform final testing, only works for predefined datasets (default=0)')
    args = parser.parse_args()

    # force this for now
    args.TESTING = 0

    # parse some arguments
    args.PATH = 'data/' + args.DATASET + '/'
    if args.NETWORK_LAYERS == 'none':
        args.NETWORK_LAYERS = []
    else:
        args.NETWORK_LAYERS = [int(x) for x in args.NETWORK_LAYERS.split(',')]

    args.EMBEDDING_LAYERS = [int(x) for x in args.EMBEDDING_LAYERS.split(',')]
    args.DROPOUT /= 100.

    # this must be enforced otherwise if no edges are found, predictions are made on nothing resulting in nan loss
    if args.FILTER == 'localpool' and args.SELF_LINKS == 1:
        sys.exit('cant use localpool with self_links=1, please use filter=efgcn, max_adjacency=0 and self_links=1')
    if args.FILTER == 'lgcn':
        args.SELF_LINKS = 1

    # print command line arguments
    if args.VERBOSE >= 1:
        for arg, value in sorted(args.__dict__.items(), key=lambda x: x[0]):
            print(arg, value)

    if args.RUNS > 0:
        test_accuracy = []
        for arg, value in sorted(args.__dict__.items(), key=lambda x: x[0]):
            print(arg + ' ' + str(value) + '\n')
        for r in xrange(args.RUNS):
            np.random.seed()
            args.RANDOM_SEED = np.random.randint(0, 1000)
            np.random.seed(args.RANDOM_SEED)
            temp_string = 'random seed: ' + str(args.RANDOM_SEED)
            print(temp_string)

            # run experiment
            test_accuracy += experiment(args)

        temp_string = str(test_accuracy)
        print(temp_string)

        mean_accuracy = np.mean(test_accuracy)
        temp_string = 'mean:\t\t' + str(np.mean(test_accuracy))
        print(temp_string)
        variance = np.var(test_accuracy)
        temp_string = 'variance:\t' + str(variance)
        print(temp_string)

        std = np.std(test_accuracy)
        temp_string = 'std:\t\t' + str(std)
        print(temp_string)

        sem = np.std(test_accuracy)/ np.sqrt(args.RUNS)
        temp_string = 'SEM:\t\t' + str(sem)
        print(temp_string)
    else:
        np.random.seed(args.RANDOM_SEED)
        experiment(args)


if __name__ == "__main__":
    main()
