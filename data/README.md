# DATA
Each dataset has its own directory
Some datasets have a predefined train and test split, this test split should not be used during training nor during hyper-parameter tuning.
The AIFB and MUTAG datasets only exist with this predefined split
The RITA dataset had versions with predefined splits and a version without, do not use the version without other than for debugging.

Each dataset contains a set of nodes, a subset of those nodes are in the train set, and a subset is in the test set. 
The nodes (and their features) are in the *.content files
The edges (and their features) are in the *.cites files

## nodes
Within the *.content file:
* column  [0]	: the first column denotes the node index
* columns [1:-2]: the second through first to last column denote the node features (if available)
* column  [-2]	: the first to last column denotes the label
* column  [-1]	: the last column denotes which split that node belongs to (0 for unlabelled, 1 for train split, 2 for test split)

## edges
Within the *.cites file the col
* column  [0]	: the first column denotes the node where the edge is originating from
* column  [1]	: the second columns denotes the destination node
* columns [1:]	: the remainder of the columns are the edge features of that edge 
