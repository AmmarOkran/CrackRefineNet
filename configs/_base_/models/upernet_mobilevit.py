# model settings
# configs/my_mobilevit_segmentation.py
norm_cfg = dict(type='SyncBN', requires_grad=True)
custom_imports = dict(imports='mmpretrain.models', allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/mobilevit/mobilevit-small_3rdparty_in1k_20221018-cb4f741c.pth'  # Specify the path to the pretrained checkpoint if available
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,  # Set to None if using a custom checkpoint
    backbone=dict(
        type='mmpretrain.MobileViT',  # Specify MobileViT from mmpretrain
        arch='small',                  # Choose MobileViT variant ('xs', 's', 'base', etc.)
        out_indices=[0, 1, 2, 3],     # Output indices for feature maps at different stages
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint_file,  # Path to pretrained MobileViT weights, if any
            prefix='backbone.'
        )
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[32, 64, 96, 128],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
