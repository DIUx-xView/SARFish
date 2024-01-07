#!/usr/bin/env python3

"""SARFish reference model training script
"""

import datetime
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import torch
import torch.utils.tensorboard

import SARModel

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    #=========================
    # Config
    #=========================

    environment_path = Path("environment.yaml")
    with open(str(environment_path), "r") as f:
        config = yaml.safe_load(f)

    # TRAIN
    trainedModelPath = Path(config["TRAIN"]["TrainedModelPath"])
    if not trainedModelPath.exists():
        trainedModelPath.mkdir()

    tensorBoardFolder = Path(config["TRAIN"]["TensorBoardFolder"])
    batchSize = int(config["TRAIN"]["BatchSize"])
    numberEpochs = int(config["TRAIN"]["NumberEpochs"])
    modelSavingFreq = int(config["TRAIN"]["ModelSavingFreq"])

    #
    # Pass the config across
    SARModel.config = config

    #
    # Tensorboard writer
    now = datetime.datetime.now()
    outDir = Path(
        tensorBoardFolder, now.strftime("%Y-%m-%d-%H-%M-%S")
    )
    if not outDir.exists():
        outDir.mkdir(parents = True, exist_ok = True)

    tb_writer = torch.utils.tensorboard.SummaryWriter(str(outDir))

    #=========================
    # MAIN
    #=========================

    (trainFileList, trainLabelDF) = SARModel.obtainLabel( 1 )
    (valFileList, valLabelDF) = SARModel.obtainLabel( 2 )
    print(f"trainFileList: {trainFileList}")
    #exit()

    #
    # to DEBUG, uncomment the if statement, and run DEBUG_train_label.py
    if 1:
        debug_path = Path("DEBUG_data", "trainFileList.npy")
        with open(str(debug_path), "wb") as f:
            np.save(f, trainFileList, allow_pickle = True)

        trainLabelDF.to_csv('DEBUG_data/trainLabelDF.csv')

    # 
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #
    print(f'77')
    train_data = SARModel.SarfishDataset(trainFileList, trainLabelDF)
    train_sampler = torch.utils.data.RandomSampler(train_data)
    train_data_loader = torch.utils.data.DataLoader(
        train_data, batch_size = batchSize, sampler = train_sampler, 
        num_workers = 0, collate_fn = collate_fn
    )

    #
    val_data = SARModel.SarfishDataset(valFileList, valLabelDF)
    val_sampler = torch.utils.data.SequentialSampler(val_data)
    val_data_loader = torch.utils.data.DataLoader(
        val_data, batch_size = batchSize, sampler = val_sampler, 
        num_workers = 0, collate_fn = collate_fn
    )

    #
    # instantiate model with a number of classes
    model = SARModel.SARFishModel( num_classes = 4 )

    # move model to the correct device
    model.to(device)

    # construct an optimizer
    numBatch = len(train_data_loader)
    val_numBatch = len(val_data_loader)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr = 1e-5, weight_decay = 0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, numBatch * numberEpochs
    ) 

#def train(
#        model: torch.nn.Module, 
#        train_data_loader: torch.utils.data.DataLoader,
#        lr_scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
#        optimizer: torch.optim.Adam,
#        tb_writer: torch.utils.tensorboard.SummaryWriter,
#    ):
#
#def test(
#        model: torch.nn.Module, 
#        val_data_loader: torch.utils.data.DataLoader,
#        tb_writer: torch.utils.tensorboard.SummaryWriter,
#    ):

    #
    tb_id = 0
    val_tb_id = 0
    for epochId in range(numberEpochs):
        checkpoint_path = Path(
            trainedModelPath, f"epoch_{epochId}.pth"
        )
        # TRAIN
        #
        model.train()
        allLoss = 0
        # DEBUG: (images, targets) = next(iter(train_data_loader))
        for bId, (images, targets) in enumerate(train_data_loader):
            print( 
                f"epochId: {epochId} Train BatchId: {bId} out of {numBatch}"
            )
            tb_writer.add_scalar(
                "lr", lr_scheduler.get_last_lr()[0], tb_id
            )
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            one_loss = losses.item()
            allLoss += one_loss
            #
            # NOTE: losses.item() is loss_dict['classification'].item() + loss_dict['bbox_regression'].item() + loss_dict['bbox_ctrness'].item()
            #
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            lr_scheduler.step()
            tb_writer.add_scalar("Loss/train", one_loss, tb_id)
            tb_id += 1
        #
        # VALIDATION
        #
        model.train()  # should really be model.eval(), but to cater for the FCOS.forward() if self.training statement
        with torch.no_grad():
            val_allLoss = 0
            for vId, (vImages, vTargets) in enumerate(val_data_loader):
                print( 
                    f"epochId: {epochId} Validation BatchId: {vId} "
                    f"out of {val_numBatch}" 
                )
                val_loss_dict = model(vImages, vTargets)
                val_losses = sum(loss for loss in val_loss_dict.values())
                val_one_loss = val_losses.item()
                val_allLoss += val_one_loss
                tb_writer.add_scalar(
                    "Loss/Validation", val_one_loss, val_tb_id
                )
                val_tb_id += 1
        #
        # SAVE MODEL
        #
        if (epochId % modelSavingFreq) == 0:
            torch.save(model.state_dict(), str(checkpoint_path))
    #
    # Save the last one
    torch.save(model.state_dict(), str(checkpoint_path))

    # Save the tensorboard
    tb_writer.close()
    print("Training complete!")

if __name__ == "__main__":
    main()
