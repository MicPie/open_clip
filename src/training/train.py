import os
import time
import json
import numpy as np

import torch
import torch.nn as nn

from torch.cuda.amp import autocast
import torch.distributed as dist

from .zero_shot import zero_shot_eval

import sys
import pdb
import wandb

import logging

def is_master(args):
    return (not args.distributed) or args.gpu == 0


def contrastive_loss(sim, infoloob=False):
    # Based on: https://github.com/HobbitLong/SupContrast/blob/master/losses.py#L11

    # for numerical stability subtract max in log space which equals division in normal space
    logits_max, _ = torch.max(sim, dim=1, keepdim=True)
    logits = sim - logits_max.detach()
    
    exp_logits = torch.exp(logits)
    mask = torch.eye(sim.shape[0], sim.shape[1], device=sim.device)
    if infoloob:
        # with infoloob we don't incorporate the self-similarity in the denominator term
        exp_logits[mask] = 0
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    loss = - (args.cl_temperature * mean_log_prob_pos).mean()
    # TO DO: Check if we need to carry out the entire calculation on every GPU or if we can split it like with the CLASP setup.
    return loss


def get_loss(model, images, texts, loss_img, loss_txt, args):
    image_features, text_features, logit_scale = model(images, texts, args.loss_type)
    logit_scale = logit_scale.mean()
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        if args.loss_type == "CLIP":
            # this is needed to send gradients back everywhere.
            logits_per_image = logit_scale * all_image_features @ all_text_features.t()
            logits_per_text = logits_per_image.t()
        elif args.loss_type == "FILIP":
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features  = torch.cat(gathered_text_features,  dim=0)
            sim_image = torch.einsum("imd,tnd->itmn", image_features, all_text_features)
            sim_text  = torch.einsum("tnd,imd->tinm", text_features, all_image_features)
            sim_image = sim_image.max(dim=3).values.mean(dim=2) # itmn, max: itm, mean: it
            sim_text  = sim_text.max(dim=3).values.mean(dim=2) # tinm, max: tin, mean: ti

    else:
        if args.loss_type == "CLIP":
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()
        elif args.loss_type == "FILIP":
            sim = torch.einsum("imd,tnd->itmn", image_features, text_features)
            sim_image = sim.max(dim=3).values.mean(dim=2)   # itmn, max: itm, mean: it
            sim_text  = sim.max(dim=2).values.mean(dim=2).T # itmn, max: itn, mean: it, transpose: ti

    if args.loss_type == "CLIP":
        ground_truth = torch.arange(len(logits_per_image)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

        total_loss = (
            loss_img(logits_per_image, ground_truth)
            + loss_txt(logits_per_text, ground_truth)
        ) / 2

    elif args.loss_type == "FILIP":
        sim_image *= logit_scale
        sim_text  *= logit_scale

        loss_image = contrastive_loss(sim_image)
        loss_text  = contrastive_loss(sim_text)
        total_loss = (loss_image + loss_text) / 2

    return total_loss


def train(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    os.environ["WDS_EPOCH"] = str(epoch)
    
    model.train()

    dataloader, sampler = data['train'].dataloader,  data['train'].sampler

    if args.loss_type == "CLIP":
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        if args.gpu is not None:
            loss_img = loss_img.cuda(args.gpu)
            loss_txt = loss_txt.cuda(args.gpu)
    elif args.loss_type == "FILIP":
        # TO DO: Make ugly hack nice.
        loss_img = None
        loss_txt = None

    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches

    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images, texts = batch
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            texts = texts.cuda(args.gpu, non_blocking=True)

        data_time = time.time() - end

        m = model.module if args.distributed or args.dp else model

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                total_loss = get_loss(model, images, texts, loss_img, loss_txt, args)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            total_loss = get_loss(model, images, texts, loss_img, loss_txt, args)
            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

        batch_time = time.time() - end
        end = time.time()

        if is_master(args) and (i % 100) == 0:
            num_samples = i * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {total_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {m.logit_scale.data:.3f}"
            )
            # save train loss / etc.

            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "loss": total_loss.item(),
                "data_time": data_time,
                "batch_time": batch_time,
                "scale":  m.logit_scale.data.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)
                if args.wandb:
                    wandb.log({name: val, 'step': timestep})


def evaluate(model, data, epoch, args, tb_writer=None, steps=None):
    if not is_master(args):
        return
    
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)

    dataloader = data['val'].dataloader

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    cumulative_loss = 0.0
    num_elements = 0.0
    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for batch in dataloader:
            images, texts = batch
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                texts = texts.cuda(args.gpu, non_blocking=True)

            image_features, text_features, logit_scale = model(images, texts)
            all_image_features.append(image_features)
            all_text_features.append(text_features)
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(len(images)).long()
            if args.gpu is not None:
                ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
            total_loss = (
                loss_img(logits_per_image, ground_truth)
                + loss_txt(logits_per_text, ground_truth)
            ) / 2

            batch_size = len(images)
            cumulative_loss += total_loss * batch_size
            num_elements += batch_size

        metrics = get_metrics(
            image_features=torch.cat(all_image_features),
            text_features=torch.cat(all_text_features),
            logit_scale=logit_scale
        )
        loss = cumulative_loss / num_elements
        metrics.update(
            **{"val_loss": loss.item(), "epoch": epoch, "num_elements": num_elements}
        )
        metrics.update(zero_shot_metrics)

        logging.info(
            f"Eval Epoch: {epoch} "
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )

        if args.save_logs:
            for name, val in metrics.items():
                if tb_writer is not None:
                    tb_writer.add_scalar(f"val/{name}", val, epoch)
        if args.wandb:
            for name, val in metrics.items():
                wandb.log({f"val/{name}": val, 'epoch': epoch})

    if args.save_logs:
        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
