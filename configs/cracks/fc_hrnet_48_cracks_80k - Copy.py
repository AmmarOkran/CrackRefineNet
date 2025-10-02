_base_ = [
    '../_base_/models/fcn_hr18.py', './data.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        num_classes=5,
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])))

data = dict(samples_per_gpu=2)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=5)
evaluation = dict(interval=1000, metric=['mIoU', 'mDice', 'mFscore'], save_best='mIoU')


