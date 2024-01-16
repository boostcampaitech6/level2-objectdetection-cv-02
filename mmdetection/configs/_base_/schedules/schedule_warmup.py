# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='Adam', lr=0.002)
# optimizer = dict(type='AdamW', lr=0.02)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.005,
    min_lr_ratio=1e-5)
runner = dict(type='EpochBasedRunner', max_epochs=12)
