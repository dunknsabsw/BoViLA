import torch
import math
import sys
from typing import Iterable
import util.misc as misc
import util.lr_sched as lr_sched
import torch.distributed as dist
import contextlib
from tqdm import tqdm
import json


def remove_duplicate_dicts(dict_list):
    seen = set()
    unique_dicts = []
    for d in dict_list:
        tuple_repr = tuple(
            sorted(
                (k, frozenset(v.items())) if isinstance(v, dict) else (k, v)
                for k, v in d.items()
            )
        )
        if tuple_repr not in seen:
            seen.add(tuple_repr)
            unique_dicts.append(d)
    return unique_dicts


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss_scaler,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = int(len(data_loader) / 4)
    accum_iter = args.accum_iter

    optimizer.zero_grad()
    total_steps = len(data_loader)
    progress_bar = tqdm(
        metric_logger.log_every(data_loader, print_freq, header),
        total=total_steps,
        desc="Processing",
    )

    for data_iter_step, data in enumerate(progress_bar):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        loss = model(data)
        task_loss_value = {}
        total_loss = 0

        for task in loss.keys():
            task_loss = loss[task]
            total_loss = total_loss + task_loss
            task_loss_value[task] = task_loss.item()

        total_loss_value = total_loss.item()

        if not math.isfinite(total_loss_value):
            print("Loss is {}, stopping training".format(total_loss_value))
            sys.exit(1)

        total_loss = total_loss / accum_iter

        loss_scaler(
            total_loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=total_loss_value)
        try:
            metric_logger.update(vqa_loss=task_loss_value["vqa"])
        except:
            metric_logger.update(edl_mse_loss=task_loss_value["edl_mse"])
            metric_logger.update(edl_reg_loss=task_loss_value["edl_reg"])

        if args.vaq:
            metric_logger.update(vaq_loss=task_loss_value["vaq"])
        if args.aqa:
            metric_logger.update(aqa_loss=task_loss_value["aqa"])

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        loss_str = ", ".join([f"{k}: {v:.3f}" for k, v in loss.items()])
        progress_bar.set_description(f"Step {data_iter_step}, Loss: {loss_str}")

    progress_bar.close()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args=None,
):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = int(len(data_loader) / 4)

    qtype_mapping = ["CH", "CW", "TN", "TC", "TP", "DL", "DC", "DO"]
    option_mapping = ["A", "B", "C", "D", "E"]
    jsonsave = []

    for data_iter_step, data in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        answer = data["answer"].cuda()
        bsz = answer.shape[0]

        with torch.no_grad():
            logits, u = model(data, inference=True)

        count = (logits != 0).sum(-1)
        prediction = (logits.sum(-1) / count).argmin(-1)

        u = torch.gather(u, 1, prediction.unsqueeze(1)).squeeze(1)  # [4]

        eval = answer == prediction
        acc = eval.sum().item() / bsz

        misc.log_qtype(data, eval, metric_logger, args)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(n=bsz, acc=acc)

        for j in range(bsz):
            try:
                item = {
                    "vid": int(data["vid"][j]),
                    "qid": int(data["qid"][j]),
                    "qtype": qtype_mapping[data["qtype"][j] - 1],
                    "pre": option_mapping[prediction[j]],
                    "ans": option_mapping[data["answer"][j]],
                    "u": u[j].cpu().item(),
                }
            except:
                item = {
                    "vid": str(data["vid"][j]),
                    "qid": int(data["qid"][j]),
                    "qtype": qtype_mapping[data["qtype"][j] - 1],
                    "pre": option_mapping[prediction[j]],
                    "ans": option_mapping[data["answer"][j]],
                    "u": u[j].cpu().item(),
                }
            jsonsave.append(item)

    # Gather all results to the rank 0 process
    gathered_results = [None for _ in range(args.world_size)]
    dist.all_gather_object(gathered_results, jsonsave)

    if misc.is_main_process():
        # Merge all results into one list
        all_results = [item for sublist in gathered_results for item in sublist]
        all_results = remove_duplicate_dicts(all_results)
        # Save to JSON
        with open(f"{args.output_dir}/epoch{epoch}.json", "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"json of epoch{epoch} is saved !")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
