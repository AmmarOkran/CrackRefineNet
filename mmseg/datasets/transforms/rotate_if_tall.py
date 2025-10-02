# mmseg/datasets/pipelines/rotate_if_tall.py
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmseg.registry import TRANSFORMS
import cv2

@TRANSFORMS.register_module()
class RotateIfTall(BaseTransform):
    # def __init__(self, size_divisor=32, interpolation=None):
    #     self.size_divisor = size_divisor
    #     self.interpolation = interpolation

    def transform(self, results: dict) -> dict:
        img = results['img']
        # print('before', img.shape)
        if img.shape[0] < img.shape[1]:
            results['img'] = np.rot90(img, k=1).copy()
            if 'gt_seg_map' in results:
                results['gt_seg_map'] = np.rot90(results['gt_seg_map'], k=1).copy()
        
        # print('after',results['img'].shape)
        return results
    # def __repr__(self):
    #     repr_str = self.__class__.__name__
    #     repr_str += (f'(size_divisor={self.size_divisor}, '
    #                  f'interpolation={self.interpolation})')
    #     return repr_str


@TRANSFORMS.register_module()
class ForceResize(BaseTransform):
    def __init__(self, size=(384, 192)):
        self.size = size

    def transform(self, results):
        h, w = self.size
        results['img'] = cv2.resize(results['img'], (w, h))
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = cv2.resize(results['gt_seg_map'], (w, h), interpolation=cv2.INTER_NEAREST)
        return results