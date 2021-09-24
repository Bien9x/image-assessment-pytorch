# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from datasets import ImageDataset

if __name__ == '__main__':
    # sanity  check dataset
    data = ImageDataset("data/images","data/AVA/ava_labels_test.json",10)
    print(data.__getitem__(0))