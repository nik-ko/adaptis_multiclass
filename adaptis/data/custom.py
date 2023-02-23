from pathlib import Path

import cv2
import numpy as np

from .base import BaseDataset


def extract_digit(n):
    # divide by 4 and extract n'th digit from right to left
    return np.vectorize(lambda x: x // 4 // 10 ** n % 10 if x > 0 else 0)


def extract_instance_ids(x):
    return extract_digit(0)(x).astype(np.int32)  # right


def extract_class_ids(x):
    return extract_digit(1)(x).astype(np.int32)  # left


class CustomDataset(BaseDataset):
    def __init__(self, dataset_path, split='train', **kwargs):
        super(CustomDataset, self).__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self.dataset_samples = []
        images_path = sorted((self.dataset_path / split).rglob('*rgb.png'))
        for image_path in images_path:
            image_path = str(image_path)
            mask_path = image_path.replace('rgb.png', 'im.png')
            self.dataset_samples.append((image_path, mask_path))

    @staticmethod
    def get_instance_mask(instance_mask):
        # Extract instance ids from instance mask and return new instance mask with ids
        instance_mask = extract_instance_ids(instance_mask)

        return instance_mask

    @staticmethod
    def get_semantic_segmentation(instance_mask):
        # Decode class mask and return a segmentation mask for classes
        segmentation_mask = extract_class_ids(instance_mask)
        return segmentation_mask

    def get_sample(self, index):
        image_path, mask_path = self.dataset_samples[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.int32)

        sample = {'image': image}
        if self.with_segmentation:
            sample['semantic_segmentation'] = self.get_semantic_segmentation(instances_mask)
        else:
            instances_mask += 1

        instances_mask = self.get_instance_mask(instances_mask)
        instances_ids = self.get_unique_labels(instances_mask, exclude_zero=True)

        instances_info = {
            x: {'class_id': 1, 'ignore': False} for x in instances_ids
        }

        sample.update({
            'instances_mask': instances_mask,
            'instances_info': instances_info,
        })

        return sample

    @property
    def stuff_labels(self):
        return [0]

    @property
    def things_labels(self):
        return [1, 2, 3, 4, 5]
