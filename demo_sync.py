import numpy as np
import datasets

dataset = datasets.load_from_disk("processed_dataset/teleband_dataset_mp3")

print(len(dataset))
for item in dataset:
    print(item)
