_base_ = [
 './data.py',
 './gated_skip_model.py',
 '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)

# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.00006,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))

# lr_config = dict(
#     _delete_=True,
#     policy='poly',
#     warmup='linear',
#     warmup_iters=1500,
#     warmup_ratio=1e-6,
#     power=1.0,
#     min_lr=0.0,
#     by_epoch=False)

data = dict(samples_per_gpu=2)
checkpoint_config = dict(by_epoch=False, interval=1500, max_keep_ckpts=5)
evaluation = dict(interval=1500, metric=['mIoU', 'mDice', 'mFscore'], save_best='mIoU')