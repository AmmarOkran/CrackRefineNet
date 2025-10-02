_base_ = [
    '../_base_/models/upernet_convnext.py', './data_cfd.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_in21k_20220301-262fd037.pth'
crop_size = (64,64)
randomness = dict(seed=42)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        drop_path_rate=0.1,
        in_channels=64,
        output_stride=16,
        # init_cfg=None
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')
    ),
    decode_head=dict(
        _delete_=True, 
        type='AFRCAMLASTSegmentationModel',
        in_channels=1024,
        channels=270,
        encoder_channels=(128, 256, 512, 1024),
        num_classes=2,
        upsampling=4,
        out_decode=256,
        threshold=0.5,
        out_channels=1,  
        loss_decode = [dict(type='CrossEntropyLoss', use_sigmoid=True, 
                         loss_name='loss_bce', loss_weight=1.0),
                       dict(type='DiceBinaryLoss', loss_name='loss_dicebinary', loss_weight=0.5),
                       ],
        ),
        auxiliary_head=None,
        )



# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001),
        # clip_grad=dict(max_norm=2, norm_type=2),
        # accumulative_counts = 4,
    # paramwise_cfg=dict(
    #     custom_keys={
    #         # 'backbone': dict(lr_mult=0.0)  # FREEZE encoder
    #         # 'absolute_pos_embed': dict(decay_mult=0.),
    #         # 'relative_position_bias_table': dict(decay_mult=0.),
    #         # 'norm': dict(decay_mult=0.)
    #     })
        )


param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=3000),
    dict(
        type='PolyLR',
        power=1.0,
        begin=3000,
        end=20000,
        eta_min=0.0,
        by_epoch=False,
    )
]

default_hooks = dict(
    # early_stopping=dict(
    #     type="EarlyStoppingHook",
    #     monitor="mIoU",
    #     patience=10,
    #     min_delta=0.005),
    checkpoint=dict(save_best='mIoU')
)

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer', alpha=0.3)

# data = dict(samples_per_gpu=2)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=5)
train_cfg = dict(val_interval=500)
# train_cfg = dict(max_iters=3000, val_interval=500)
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_cfg = dict(type='TestLoop')