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
class PlantSeg(BaseSegDataset):

    METAINFO = dict(
        classes=(
            'Background','apple black rot','apple mosaic virus','apple rust','apple scab','banana anthracnose',
            'banana black leaf streak','banana bunchy top','banana cigar end rot','banana cordana leaf spot','banana panama disease',
            'basil downy mildew','bean halo blight','bean mosaic virus','bean rust','bell pepper bacterial spot',
            'bell pepper blossom end rot','bell pepper frogeye leaf spot','bell pepper powdery mildew','blueberry anthracnose',
            'blueberry botrytis blight','blueberry mummy berry','blueberry rust','blueberry scorch','broccoli alternaria leaf spot',
            'broccoli downy mildew','broccoli ring spot','cabbage alternaria leaf spot','cabbage black rot','cabbage downy mildew',
            'carrot alternaria leaf blight','carrot cavity spot','carrot cercospora leaf blight','cauliflower alternaria leaf spot',
            'cauliflower bacterial soft rot','celery anthracnose','celery early blight','cherry leaf spot','cherry powdery mildew',
            'citrus canker','citrus greening disease','coffee berry blotch','coffee black rot','coffee brown eye spot','coffee leaf rust',
            'corn gray leaf spot','corn northern leaf blight','corn rust','corn smut','cucumber angular leaf spot','cucumber bacterial wilt',
            'cucumber powdery mildew','eggplant cercospora leaf spot','eggplant phomopsis fruit rot','eggplant phytophthora blight',
            'garlic leaf blight','garlic rust','ginger leaf spot','ginger sheath blight','grape black rot','grape downy mildew','grape leaf spot',
            'grapevine leafroll disease','lettuce downy mildew','lettuce mosaic virus','maple tar spot','peach anthracnose','peach brown rot',
            'peach leaf curl','peach rust','peach scab','plum bacterial spot','plum brown rot','plum pocket disease','plum pox virus','plum rust',
            'potato early blight','potato late blight','raspberry fire blight','raspberry gray mold','raspberry leaf spot','raspberry yellow rust',
            'rice blast','rice sheath blight','soybean bacterial blight','soybean brown spot','soybean downy mildew','soybean frog eye leaf spot',
            'soybean mosaic', 'soybean rust', 'squash powdery mildew','strawberry anthracnose','strawberry leaf scorch','tobacco blue mold','tobacco brown spot',
            'tobacco frogeye leaf spot','tobacco mosaic virus','tomato bacterial leaf spot','tomato early blight','tomato late blight','tomato leaf mold',
            'tomato mosaic virus','tomato septoria leaf spot','tomato yellow leaf curl virus','wheat bacterial leaf streak (black chaff)','wheat head scab',
            'wheat leaf rust','wheat loose smut','wheat powdery mildew','wheat septoria blotch','wheat stem rust','wheat stripe rust','zucchini bacterial wilt',
            'zucchini downy mildew','zucchini powdery mildew','zucchini yellow mosaic virus'
            ),
        palette=[
            [0, 0, 0], [141, 197, 244], [39, 123, 120], [107, 128, 9], [83, 206, 91], [191, 66, 143],
            [255, 236, 184], [151, 31, 20], [255, 150, 77], [61, 233, 244], [165, 192, 120],
            [102, 226, 255], [41, 175, 238], [191, 162, 225], [155, 104, 28], [242, 41, 31],
            [20, 241, 55], [100, 9, 210], [216, 120, 19], [86, 42, 134], [156, 210, 77],
            [110, 92, 228], [255, 216, 106], [217, 51, 157], [171, 35, 219], [54, 168, 207],
            [233, 173, 52], [247, 205, 219], [29, 148, 222], [153, 221, 65], [208, 175, 112],
            [33, 204, 71], [231, 122, 90], [216, 125, 142], [91, 255, 167], [229, 100, 182],
            [198, 157, 140], [181, 42, 109], [29, 205, 158], [182, 251, 93], [237, 146, 119],
            [247, 79, 217], [62, 185, 165], [207, 61, 242], [116, 145, 241], [241, 229, 140],
            [218, 105, 69], [195, 164, 85], [142, 32, 198], [247, 124, 4], [61, 255, 161],
            [203, 105, 92], [229, 249, 73], [147, 209, 111], [87, 236, 183], [120, 166, 217],
            [76, 143, 247], [247, 211, 5], [231, 49, 125], [179, 132, 183], [83, 255, 124],
            [246, 31, 77], [250, 120, 228], [208, 238, 49], [231, 114, 169], [75, 227, 208],
            [145, 247, 174], [66, 215, 73], [233, 242, 95], [137, 163, 93], [247, 111, 102],
            [181, 255, 69], [200, 157, 41], [254, 105, 167], [119, 242, 159], [197, 81, 235],
            [245, 170, 89], [255, 145, 205], [122, 224, 255], [250, 182, 28], [199, 96, 152],
            [218, 79, 221], [246, 214, 94], [212, 39, 81], [108, 255, 215], [249, 247, 44],
            [132, 216, 137], [175, 75, 70], [204, 94, 228], [228, 191, 174], [147, 245, 71],
            [216, 35, 136], [234, 184, 224], [188, 210, 103], [244, 69, 52], [112, 244, 186],
            [159, 241, 134], [214, 140, 86], [242, 151, 57], [221, 149, 255], [156, 85, 179],
            [222, 123, 51], [102, 244, 255], [249, 189, 180], [185, 251, 151], [227, 87, 35],
            [171, 255, 47], [251, 151, 242], [95, 214, 78], [224, 243, 152], [215, 137, 35],
            [244, 106, 149], [203, 191, 235], [167, 247, 133], [232, 97, 79], [213, 63, 156],
            # [166, 255, 114], [251, 219, 62], [207, 131, 161], [115, 201, 255], [252, 155, 104]
    ] 
        )
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
    

    # def load_data_list(self) -> List[dict]:
    #     """Load annotation from directory or annotation file.

    #     Returns:
    #         list[dict]: All data info of dataset.
    #     """
    #     data_list = []
    #     img_dir = self.data_prefix.get('img_path', None)
    #     ann_dir = self.data_prefix.get('seg_map_path', None)
    #     ann_dir2 = self.data_prefix.get('stweak_map_path', None)
    #     if not osp.isdir(self.ann_file) and self.ann_file:
    #         assert osp.isfile(self.ann_file), \
    #             f'Failed to load `ann_file` {self.ann_file}'
    #         lines = mmengine.list_from_file(
    #             self.ann_file, backend_args=self.backend_args)
    #         for line in lines:
    #             img_name = line.strip()
    #             data_info = dict(
    #                 img_path=osp.join(img_dir, img_name + self.img_suffix))
    #             if ann_dir is not None:
    #                 seg_map = img_name + self.seg_map_suffix
    #                 data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
    #             data_info['label_map'] = self.label_map
    #             data_info['reduce_zero_label'] = self.reduce_zero_label
    #             data_info['seg_fields'] = []
    #             data_list.append(data_info)
    #     else:
    #         _suffix_len = len(self.img_suffix)
    #         for img in fileio.list_dir_or_file(
    #                 dir_path=img_dir,
    #                 list_dir=False,
    #                 suffix=self.img_suffix,
    #                 recursive=True,
    #                 backend_args=self.backend_args):
    #             data_info = dict(img_path=osp.join(img_dir, img))
    #             if ann_dir is not None:
    #                 seg_map = img[:-_suffix_len] + self.seg_map_suffix
    #                 data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
    #                 data_info['stweak_map_path'] = osp.join(ann_dir2, seg_map)
    #             data_info['label_map'] = self.label_map
    #             data_info['reduce_zero_label'] = self.reduce_zero_label
    #             data_info['seg_fields'] = []
    #             data_list.append(data_info)
    #         data_list = sorted(data_list, key=lambda x: x['img_path'])
    #     return data_list
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
                    # data_info['stweak_map_path'] = osp.join(ann_dir2, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_info['has_strong'] = self.define_strong(data_info['seg_map_path'])
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
    
    
