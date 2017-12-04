import pandas as pd

df = pd.read_csv('cora.content', header=None, delimiter='\t')

with open('cora.cites', 'r') as f:
    with open('cora_plus.cites', 'w') as fp:
        for line in f.readlines():
            node_features = df.loc[df[0] == int(line.split('\t')[0])].as_matrix()[0][1:-1]
            fp.write(line[:-1] + '\t' + '\t'.join([str(x) for x in node_features]) + '\n')
