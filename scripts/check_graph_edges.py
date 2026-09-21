import sys
sys.path.append("/data1/schulz/MRF-DIFUSCO")

import difusco
from difusco.co_datasets.mis_dataset import MISDataset

dataset_file = "/data1/schulz/MRF-DIFUSCO-data/mis_er50_5k/train/*.gpickle"

dataset = MISDataset(dataset_file)

_, graph_data, _ = dataset[0]

u = graph_data.edge_index[0]
v = graph_data.edge_index[1]

print("all edges:", u.numel())
print("non-self:", (u != v).sum().item())
print("canonical:", (u < v).sum().item())

