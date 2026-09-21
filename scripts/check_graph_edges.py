import difusco
from difusco.co_datasets.mis_dataset import MISDataset

dataset_file = "/data1/schulz/MRF-DIFUSCO-data/mis_er50_5k/train/*.gpickle"

dataset = MISDataset(dataset_file)

print(dataset[0])

