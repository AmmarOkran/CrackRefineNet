# Copyright (c) OpenMMLab. All rights reserved.
"""Use the pytorch-grad-cam tool to visualize Class Activation Maps (CAM).

requirement: pip install grad-cam
"""

from argparse import ArgumentParser
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mmengine import Config
from mmengine.model import revert_sync_batchnorm
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image

from mmseg.apis import inference_model, init_model, show_result_pyplot
from mmseg.utils import register_all_modules
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable



def save_overlay_clean(img_rgb, cam_gray, save_path='cam_proposed_colorbar.png',
                       cmap='jet', alpha=0.4, vmin=None, vmax=None):
    """
    Overlay CAM heatmap on RGB image with clean colorbar (no axes, no legend text),
    and save to file.
    """
    # Normalize image
    if img_rgb.dtype == np.uint8:
        base = img_rgb.astype(np.float32) / 255.0
    else:
        base = np.clip(img_rgb.astype(np.float32), 0, 1)

    # Normalize CAM
    if vmin is None: vmin = float(np.nanmin(cam_gray))
    if vmax is None: vmax = float(np.nanmax(cam_gray))
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(base)
    im = ax.imshow(cam_gray, cmap=cmap, norm=norm, alpha=alpha)
    ax.axis('off')  # remove axes completely

    # Colorbar with NO label
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.046,
        pad=0.04
    )
    # Remove label
    cbar.set_label("")  
    # Optional: remove ticks too if you want it 100% clean
    # cbar.set_ticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)


class SemanticSegmentationTarget:
    """wrap the model.

    requirement: pip install grad-cam

    Args:
        category (int): Visualization class.
        mask (ndarray): Mask of class.
        size (tuple): Image size.
    """

    def __init__(self, category, mask, size):
        self.category = category
        self.mask = torch.from_numpy(mask)
        self.size = size
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        model_output = torch.unsqueeze(model_output, dim=0)
        model_output = F.interpolate(
            model_output, size=self.size, mode='bilinear')
        model_output = torch.squeeze(model_output, dim=0)

        return (model_output[self.category, :, :] * self.mask).sum()


def main():
    parser = ArgumentParser() 
    # parser.add_argument('--img', default='./comment6/20160222_115305_641_721.jpg', help='Image file')
    # parser.add_argument('--config', default='./configs/crack_new_whole_rcfd/dual_deeplabv3++_link_convnext_AdamW_0.0001_0.0005dec_3000warm_crackmaster.py', help='Config file')
    # parser.add_argument('--checkpoint', default='E:/RCFD_dataset/crackmaster_Adam_0.0001_0.0001dec_3000warm/best_mIoU_iter_19500.pth', help='Checkpoint file')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-file',
        default='prediction.png',
        help='Path to output prediction file')
    parser.add_argument(
        '--cam-file', default='vis_cam.png', help='Path to output cam file')
    parser.add_argument(
        '--target-layers',
        # default='backbone.stages[-2]',  
        # default='backbone.layer4[2]',
        # default='decode_head.afr2',
        # default='decode_head.decoder2',
        # default='decode_head.conv_seg', # U-Net Mobilenetv3 SwinT
        # default='decode_head.fuse.weight_sum', # HED
        # default='decode_head.fuse_conv', # DeepCrack
        # default='decode_head.final', # CrackFormerII
        # default='decode_head.out', # CT-Crackseg
        # default='decode_head.lastlayer[6]', # LMM 
        # default='decode_head.last_block', # Effiecintcracknet 
        # default='decode_head[1].segmentation_head[0]', # CrackMaster
        default='decode_head.segmentation_head', # CrackRefineNet
        help='Target layers to visualize CAM')
    parser.add_argument(
        '--category-index', default='0', help='Category to visualize CAM')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    register_all_modules()
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # test a single image
    result = inference_model(model, args.img)

    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        draw_gt=False,
        show=False if args.out_file is not None else True,
        out_file=args.out_file)

    # result data conversion
    prediction_data = result.pred_sem_seg.data
    pre_np_data = prediction_data.cpu().numpy().squeeze(0)

    target_layers = args.target_layers
    target_layers = [eval(f'model.{target_layers}')]

    category = int(args.category_index)
    mask_float = np.float32(pre_np_data == category)

    # data processing
    image = np.array(Image.open(args.img).convert('RGB'))
    height, width = image.shape[0], image.shape[1]
    rgb_img = np.float32(image) / 255
    config = Config.fromfile(args.config)
    image_mean = config.data_preprocessor['mean']
    image_std = config.data_preprocessor['std']
    input_tensor = preprocess_image(
        rgb_img,
        mean=[x / 255 for x in image_mean],
        std=[x / 255 for x in image_std])

    # Grad CAM(Class Activation Maps)
    # Can also be LayerCAM, XGradCAM, GradCAMPlusPlus, EigenCAM, EigenGradCAM
    targets = [
        SemanticSegmentationTarget(category, mask_float, (height, width))
    ]
    with GradCAM(
            model=model,
            target_layers=target_layers) as cam: # use_cuda=torch.cuda.is_available()
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)

        if not isinstance(grayscale_cam, np.ndarray):
            grayscale_cam = grayscale_cam.cpu().numpy()

        # normalize to [0,255]
        cam_u8 = (np.clip(grayscale_cam, 0, 1) * 255).astype(np.uint8)

        # save as grayscale PNG
        cv2.imwrite("cam_attention.png", cam_u8)

        if True:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        # Create a figure and axis to plot the image and colorbar
        fig, ax = plt.subplots()
        # fig.patch.set_facecolor('k')
        ax.set_facecolor('k')
        ax.axis('off')
        # cax = ax.imshow(heatmap/255, cmap='jet')
        cax = ax.imshow(heatmap/255, cmap='jet')
        # cbar = fig.colorbar(cax)
        # Remove the axes
        ax.axis('off')
        fname=args.cam_file.split('.')[0].split('/')[-1]
        plt.savefig(f'Feat_heatmap_with_colorbar_{fname}.png')
        # plt.savefig(f'Segmentation_heatmap_with_colorbar_{fname}.png')
        # Image.fromarray(heatmap).save(f'heatmap_with_colorbar_{fname}.png')

        # save cam file
        Image.fromarray(cam_image).save(args.cam_file)
        save_overlay_clean(rgb_img, grayscale_cam, cmap='jet', alpha=0.3, vmin=0, vmax=1)


if __name__ == '__main__':
    main()
