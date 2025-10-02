# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoderTrans',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='TransMUNet',
        depth=4,),
    decode_head=dict(
        type='TransHead',
        in_channels=2048,
        # in_index=3,
        channels=512,
        out_channels=1,
        # num_convs=2,
        # concat_input=True,
        # dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        # align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=None,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
