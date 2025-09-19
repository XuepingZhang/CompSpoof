fs = 16000
chunk_len = 4  # (s)
chunk_size = chunk_len * fs
epochs=100
batch_size=64

# data configure:

# train_data = '/SMIIPdata2/zxp/composition_antispoofing/labels/full/train_label1.txt'
# dev_data = '/SMIIPdata2/zxp/composition_antispoofing/labels/full/dev_label1.txt'
# eval_data = '/SMIIPdata2/zxp/composition_antispoofing/labels/full/test_label1.txt'
# data_root = '/SMIIPdata2/zxp/composition_antispoofing/dataset'
# checkpoint=f'/SMIIPdata2/zxp/composition_antispoofing/Unet/weight/batch_{batch_size}_epoch_{epochs}'
# joint=False
# if joint:
#     checkpoint = f'/SMIIPdata2/zxp/composition_antispoofing/Unet/weight/batch_{batch_size}_epoch_{epochs}_joint'
#     start_end = 3
# else:
#     checkpoint = f'/SMIIPdata2/zxp/composition_antispoofing/Unet/weight/batch_{batch_size}_epoch_{epochs}_no_joint'
#     start_end = epochs

train_data = '/work/xz464/labels/full/train_label1.txt'
dev_data = '/work/xz464/labels/full/dev_label1.txt'
eval_data = '/work/xz464/labels/full/test_label1.txt'
data_root = '/work/xz464/dataset'
checkpoint=f'/work/xz464/composition_antispoofing_dkucc/Unet/weight/batch_{batch_size}_epoch_{epochs}'
joint=True
if joint:
    checkpoint = f'/work/xz464/composition_antispoofing_dkucc/Unet/weight/batch_{batch_size}_epoch_{epochs}_joint5'
    start_end = 5
else:
    checkpoint = f'/work/xz464/composition_antispoofing_dkucc/Unet/weight/batch_{batch_size}_epoch_{epochs}_no_joint'
    start_end = epochs

# trainer config

unet_conf = {

}
aasist_conf = {
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
}

adam_kwargs = {
    "lr": 1e-3,
    "weight_decay": 1e-5,
}
trainer_conf = {
    "optimizer": "adam",
    "optimizer_kwargs": adam_kwargs,
    "min_lr": 1e-8,
    "no_impr": 10,
    "factor": 0.5,
    "logging_period": 200  # batch number
}