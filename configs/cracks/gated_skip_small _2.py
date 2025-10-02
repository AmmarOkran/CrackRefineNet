_base_ = [
 './data.py',
 './gated_skip_model.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

data = dict(samples_per_gpu=2)
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(interval=500, metric=['mIoU', 'mDice', 'mFscore'], save_best='mIoU')