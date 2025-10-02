# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

from typing import Callable, Dict, List, Optional, Sequence, Union
import os.path as osp
import mmengine
import mmengine.fileio as fileio
import torch
import os
@DATASETS.register_module()
class CrackDatasetBinaryStweak(BaseSegDataset):

    METAINFO = dict(
        classes=('Background', 'Crack'),
        palette=[[0, 0, 0], [255,255,255]])
    # CLASSES = ('Background', 'Crack')

    # PALETTE = [[0, 0, 0], [255,255,255]]

    # def __init__(self, **kwargs):
    #     super(CrackDatasetBinary, self).__init__(
    #         reduce_zero_label=False,
    #         **kwargs)
    #     pass

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
    

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        ann_dir2 = self.data_prefix.get('stweak_map_path', None)
        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), \
                f'Failed to load `ann_file` {self.ann_file}'
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            _suffix_len = len(self.img_suffix)
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    seg_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                    data_info['stweak_map_path'] = osp.join(ann_dir2, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_info['has_strong'] = self.define_strong(data_info['seg_map_path'])
                data_info['has_weak'] = self.define_weak(data_info['stweak_map_path'])
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list
    
    def define_strong(self, strong_mask_path):
        """Define strong augmentation for the dataset.

        Args:
            data_info (dict): Data info of the dataset.

        Returns:
            dict: Strong augmentation for the dataset.
        """
        has_strong = torch.tensor(0.0)
        if self.data_prefix.get('seg_map_path', None):
            strong_mask_path = os.path.join(strong_mask_path)
            if os.path.exists(strong_mask_path):
                # strong_mask = Image.open(strong_mask_path).convert("L")
                # Convert mask to NumPy array
                # mask_np = np.array(strong_mask)
                # Check and apply threshold only if needed
                # if not set(np.unique(mask_np)).issubset({0, 255}):
                #     # print("Non-binary mask detected. Applying threshold...")
                #     _, binary_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
                #     strong_mask = Image.fromarray(binary_np)  # Replace original strong_mask with binary version
                has_strong = torch.tensor(1.0)
        return has_strong
    

    def define_weak(self, weak_mask_path):
        """Define strong augmentation for the dataset.

        Args:
            data_info (dict): Data info of the dataset.

        Returns:
            dict: Strong augmentation for the dataset.
        """
        has_weak = torch.tensor(0.0)
        if self.data_prefix.get('seg_map_path', None):
            weak_mask_path = os.path.join(weak_mask_path)
            if os.path.exists(weak_mask_path):
                # strong_mask = Image.open(strong_mask_path).convert("L")
                # Convert mask to NumPy array
                # mask_np = np.array(strong_mask)
                # Check and apply threshold only if needed
                # if not set(np.unique(mask_np)).issubset({0, 255}):
                #     # print("Non-binary mask detected. Applying threshold...")
                #     _, binary_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
                #     strong_mask = Image.fromarray(binary_np)  # Replace original strong_mask with binary version
                has_weak = torch.tensor(1.0)
        return has_weak
    
    
