# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class CrackDatasetBinaryOut(BaseSegDataset):

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
    
    
