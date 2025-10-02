_base_ = [
 '../_base_/models/upernet_mobilevit.py', './data.py',
 '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/mobilevit/mobilevit-small_3rdparty_in1k_20221018-cb4f741c.pth'
crop_size = (64,64)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=2,
    ),
    auxiliary_head=None,
        )

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0005),
    paramwise_cfg=dict(
        custom_keys={
            # 'backbone.downsample_layers.0':dict(lr_mult=3, decay_mult=0),
            # 'backbone.downsample_layers.1':dict(lr_mult=3, decay_mult=0),
            # 'backbone.stages.0': dict(lr_mult=3, decay_mult=0),
            # 'backbone.stages.1': dict(lr_mult=3, decay_mult=0),
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))


# param_scheduler = [
#     # Use a linear warm-up at [0, 100) iterations
#     dict(type='LinearLR',
#          start_factor=1e-6,
#          by_epoch=False,
#          begin=0,
#          end=3000),
#     # Use a cosine learning rate at [100, 900) iterations
#     dict(type='CosineAnnealingLR',
#          T_max=39000,
#          by_epoch=False,
#          begin=3000,
#          end=40000)
# ]

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
    checkpoint=dict(save_best='mIoU')
)

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# data = dict(samples_per_gpu=2)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=5)
train_cfg = dict(val_interval=500)
val_cfg = dict(type='ValLoop')
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_cfg = dict(type='TestLoop')
