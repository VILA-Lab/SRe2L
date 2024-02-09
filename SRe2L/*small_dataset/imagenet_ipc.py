import collections

import numpy as np
from torchvision.datasets import ImageFolder


class ImageFolderIPC(ImageFolder):
    def __init__(self, ipc=None, random_select=False, **kwargs):
        self.ipc = ipc
        if not isinstance(ipc, int) or ipc <= 0:
            raise ValueError("ipc must be specified as an integer greater than 0.")
        super(ImageFolderIPC, self).__init__(**kwargs)

        if random_select:
            self.samples, self.targets = self.random_select()
        else:
            self.samples, self.targets = self.select()

        if len(self.samples) != len(self.targets):
            raise ValueError(f"There are some classes that do not have enough ipc={self.ipc} samples.")

        self.imgs = self.samples

    def select(self):
        new_samples = []
        new_targets = []

        for class_idx in range(len(self.class_to_idx)):
            new_targets.extend([class_idx] * self.ipc)

        class_counts = collections.defaultdict(int)
        for path, class_idx in self.samples:
            if class_counts[class_idx] < self.ipc:
                class_counts[class_idx] += 1
                new_samples.append((path, class_idx))

        return new_samples, new_targets

    def random_select(self):
        new_samples = []
        new_targets = []

        for class_idx in range(len(self.class_to_idx)):
            new_targets.extend([class_idx] * self.ipc)

        class_counts = collections.defaultdict(int)
        for _, class_idx in self.samples:
            class_counts[class_idx] += 1

        class_begin_idx = 0
        for class_idx, count in class_counts.items():
            class_end_idx = class_begin_idx + count
            selected = np.random.choice(range(class_begin_idx, class_end_idx), self.ipc, replace=False)
            new_samples.extend([self.samples[i] for i in selected])
            class_begin_idx = class_end_idx

        return new_samples, new_targets


if __name__ == "__main__":
    dataset = ImageFolderIPC(root="/path/to/imagenet/train", ipc=50)
    print(len(dataset))  # list
    print(dataset[0])
