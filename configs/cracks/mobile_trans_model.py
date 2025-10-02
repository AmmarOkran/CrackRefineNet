norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MobileViT', 
        num_classes =32),
    decode_head=dict(
        type='ConvHead',
        in_channels=32,
        num_classes=5,),
    # model training and testing settings
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    )
