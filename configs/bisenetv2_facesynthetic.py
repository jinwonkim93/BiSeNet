
cfg = dict(
    model_type='bisenetv2',
    n_cats=19,
    num_aux_heads=4,
    lr_start=5e-3,
    weight_decay=1e-4,
    warmup_iters=1000,
    max_iter=160000,
    dataset='FacesyntheticDataset',
    im_root='./datasets/facesynthetic',
    train_im_anns='./datasets/facesynthetic/train.txt',
    val_im_anns='./datasets/facesynthetic/val.txt',
    scales=[0.75, 2.],
    cropsize=[512, 512],
    eval_crop=[512, 512],
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    ims_per_gpu=2,
    eval_ims_per_gpu=2,
    use_fp16=True,
    use_sync_bn=True,
    respth='./res',
)
