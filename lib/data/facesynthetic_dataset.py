#!/usr/bin/python
# -*- encoding: utf-8 -*-


import lib.data.transform_cv2 as T
from lib.data.base_dataset import BaseDataset


class FacesyntheticDataset(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(FacesyntheticDataset, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.lb_ignore = 255

        self.to_tensor = T.ToTensor(
            mean=(0.485, 0.456, 0.406), # rgb
            std=(0.229, 0.224, 0.225),
        )


