# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import subprocess
from hashlib import sha256

import torch

BLOCK_SIZE = 128 * 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('--in_file', default='D:/mmsegmentation/work_finalized_YCD/dual_deeplabv3++_link_convnext_AdamW_0.0001_0.0005dec_3000warm_YCD_ce_512x512/best_mIoU_test.pth',help='input checkpoint filename')
    parser.add_argument('--out_file', default='D:/mmsegmentation/work_finalized_YCD/pub.pth', help='output checkpoint filename')
    args = parser.parse_args()
    return args


def sha256sum(filename: str) -> str:
    """Compute SHA256 message digest from a file."""
    hash_func = sha256()
    byte_array = bytearray(BLOCK_SIZE)
    memory_view = memoryview(byte_array)
    with open(filename, 'rb', buffering=0) as file:
        for block in iter(lambda: file.readinto(memory_view), 0):
            hash_func.update(memory_view[:block])
    return hash_func.hexdigest()


def process_checkpoint(in_file, out_file):
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    torch.save(checkpoint, out_file)
    sha = sha256sum(in_file)
    final_file = out_file.rstrip('.pth') + f'-{sha[:8]}.pth'
    subprocess.Popen(['mv', out_file, final_file])


def main():
    args = parse_args()
    process_checkpoint(args.in_file, args.out_file)


if __name__ == '__main__':
    main()
