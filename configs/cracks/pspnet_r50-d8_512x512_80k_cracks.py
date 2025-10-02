_base_ = [
    '../_base_/models/pspnet_r50-d8.py', './data_croped.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))


data = dict(samples_per_gpu=2)
checkpoint_config = dict(by_epoch=False, interval=1500, max_keep_ckpts=5)
evaluation = dict(interval=1500, metric=['mIoU', 'mDice', 'mFscore'], save_best='mIoU')
