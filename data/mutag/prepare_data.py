from __future__ import print_function
import rdflib as rdf
import pandas as pd
import numpy as np

g = rdf.Graph()
g.parse('mutag_stripped.nt', format='nt')

rels = list(set(g.predicates()))
print('total relations', len(rels))
nodes = list(g.all_nodes())
print('total nodes', len(nodes))
print('total edges', len(g))

print('writing to mutag.relations...')
with open('mutag.relations', 'wb') as fr:
    for i, rel in enumerate(rels):
        fr.write(str(i) +'\t'+ rel + '\n')

train_labels = pd.read_csv('trainingSet.tsv', delimiter='\t')
test_labels = pd.read_csv('testSet.tsv', delimiter='\t')

print('writing to mutag.content...')
with open('mutag.content', 'wb') as f:
    for i, n in enumerate(nodes):
        traintest = 0
        label = 0
        train_row = train_labels.loc[train_labels['bond'] == str(n.decode('utf-8'))]
        test_row  = test_labels.loc[test_labels['bond'] == str(n.decode('utf-8'))]
        if len(train_row) > 0:
            label = int(train_row['label_mutagenic'].as_matrix()[0])
            traintest = 1
        elif len(test_row) > 0:
            label = int(test_row['label_mutagenic'].as_matrix()[0])
            traintest = 2

        f.write(str(i) + '\t' + str(label) + '\t' + str(traintest) + '\n')

print('writing to mutag.cites...')
with open('mutag.cites', 'wb') as fc:
    graph = {}
    for i, (sub, pre, obj) in enumerate(g):
        s = nodes.index(sub)
        o = nodes.index(obj)
        p = rels.index(pre)
        graph[s] = graph.get(s, {})
        graph[s][o] = graph[s].get(o, []) + [p]
        if (i % (len(g)/100) == 0 and i > 0) or i == len(g) - 1:
            print(i, len(g))
    edge_counter = 0
    for origin, destinations in graph.iteritems():
        for destination, relations in destinations.iteritems():
            one_hot_relations = np.zeros(len(rels), dtype='int32')
            one_hot_relations[relations] = 1
            fc.write(str(origin) + '\t' + str(destination) + '\t' + '\t'.join([str(x) for x in one_hot_relations]) + '\n')
            edge_counter += 1
    print('total edges with edge features', edge_counter)
print('done')
