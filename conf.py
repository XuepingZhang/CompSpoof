fs = 16000
chunk_len = 4  # (s)
chunk_size = chunk_len * fs
epochs=100
batch_size=64

joint=True # joint learning setting

train_data = 'train_label.txt'
dev_data = 'dev_label.txt'
eval_data = 'test_label.txt'
data_root = '/your_path'
checkpoint=f'/your_path/weight/batch_{batch_size}_epoch_{epochs}'

if joint:
    start_end = 5 # joint learning from epoch 5
    checkpoint = f'/your_path/weight/batch_{batch_size}_epoch_{epochs}_joint_{start_end}'
else:
    checkpoint = f'/your_path/weight/batch_{batch_size}_epoch_{epochs}_no_joint'
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