import ml_collections
import torch
from torch import nn
import datetime

def get_config():
    config = ml_collections.ConfigDict()

    # dataset config
    config.csv_path = "/root/autodl-tmp/train.csv"
    config.img_folder = "/root/autodl-tmp/train_thumbnails"
    config.aug_imgs_folder = "/root/autodl-tmp/train_thumbnails_rand_augs/train_thumbnails_rand_augs"
    config.img_size = (224, 224)
    config.aug_imgs_num = 4
    config.split = 0.8
    # mode select:
    # use_origin_datasets, use_augs_datasets, use_fuse_datasets
    config.mode = "use_fuse_datasets"

    # dataset transforms config
    config.use_augmixDataset = False
    config.use_randEraseing = True
    # mean: (0.4475443181411984, 0.4291305381177109, 0.44748035300495914), std: (0.3659115425619272, 0.34004004154735845, 0.3640951663958091)
    config.mean = [0.4475443181411984, 0.4291305381177109, 0.44748035300495914]
    config.std = [0.3659115425619272, 0.34004004154735845, 0.3640951663958091]

    # dataloader
    config.num_workers = 4
    config.pin_memory = True

    # mixup and cutmix config
    config.use_mixup = False
    config.mixup_alpha = 0.8
    config.cutmix_alpha = 0.1
    config.cutmix_minmax = None
    config.prob = 0.5
    config.switch_prob = 0.5
    config.mixup_mode = None


    # model config
    config.label_smoothing = 0.1
    config.num_classes = 5

    # lr_scheduler config
    config.use_cosine = False
    config.warmup_epoches = 0
    config.decay_epoches = 0
    config.mix_lr = 1e-5
    config.warmup_lr = 1e-5

    # training config
    config.model_name = "convnext_base"
    config.use_labelsmooth_loss = False
    config.train_epoches  = 50
    config.batch_size = 32
    config.lr = 1e-4
    config.weight_decay = 2e-4
    # select from:
    # adam, adamw,
    config.optimizer = "adam"
    config.momentum = 4e-3

    #save
    config.version = "timm_2"
    config.save_model_name = f"{config.model_name}--{datetime.date.today()}--epoch{config.train_epoches}--ver{config.version}.pt"
    config.logger_name = f"{config.model_name}--{datetime.date.today()}--epoch{config.train_epoches}--ver{config.version}.log"
    return config


if __name__ == "__main__":
    config = get_config()
    print(config)

