import torch
from torch.utils.data import DataLoader

import numpy as np
import time
import sys
import os
import time
TIME=time.strftime("%m-%d-%H_%M_%S", time.localtime())
print(TIME)
from models import utils, caption
from datasets import coco
from configuration import Config
from engine import train_one_epoch, evaluate


def main(config):
    print(config)
    device = torch.device(config.device)

    # 固定随机种子
    seed = config.seed + utils.get_rank()
    utils.set_global_random_seed(seed)

    # 生成model, loss, optimizer和scheduler
    model, CEloss, CLoss = caption.build_model(config)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop,gamma=0.9)  # 每过config.lr_drop个epoch, lr*=0.9


    # 生成数据集
    dataset_train = coco.build_dataset(config, mode='training')
    dataset_val = coco.build_dataset(config, mode='validation')
    print(f"Train: {len(dataset_train)}")
    print(f"Valid: {len(dataset_val)}")

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers)
    data_loader_val = DataLoader(dataset_val, config.batch_size,
                                 sampler=sampler_val, drop_last=False, num_workers=config.num_workers)


    # 载入checkpoint
    if os.path.exists(config.checkpoint):
        print("Loading Checkpoint...")
        checkpoint = torch.load(config.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.start_epoch = checkpoint['epoch'] + 1

    print("Start Training..")
    base_dev_loss = 1e10
    for epoch in range(config.start_epoch, config.epochs):
        model_save_path = os.path.join(config.checkpoint_save_folder,"ckpt_T{}_epo{}.pth".format(TIME,epoch+1))
        print(model_save_path)
        config.checkpoint = model_save_path
        print(f"Epoch: {epoch}")
        epoch_loss = train_one_epoch(
            config,model, CEloss,CLoss, data_loader_train, optimizer, device, epoch, config.clip_max_norm)
        lr_scheduler.step()
        print(f"Training Loss: {epoch_loss}")

        

        validation_loss = evaluate(config,model, CEloss,CLoss, data_loader_val, device)

        # 保存模型
        if validation_loss<base_dev_loss:
            base_dev_loss = validation_loss
            print("Save the best model!")
            torch.save({
                'config':str(config),
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, config.checkpoint)
            print("saved to: ",config.checkpoint)
        print(f"Validation Loss: {validation_loss}")

        print()


if __name__ == "__main__":
    config = Config()
    main(config)
