import argparse
import json
import os
from functools import partial

import torch
from layer import create_tdcl_block
from snntorch import surrogate
from tqdm import tqdm
from util.classifier_training import train_classifier
from util.dataloading import create_shd_dataloaders
from util.metric import MultiMetricBuilder
from util.tracer import Tracer

from tdcl import FullOnlineTDCLLayerTrainer, FullOnlineTDCLTrainer


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("-gf", "--gain_fast", required=True, type=float, default=1.1816)
    parser.add_argument("-gs", "--gain_slow", required=True, type=float, default=-0.8714)
    parser.add_argument("-bf", "--beta_fast", required=True, type=float, default=0.85)
    parser.add_argument("-bs", "--beta_slow", required=True, type=float, default=0.90)
    parser.add_argument("-num", "--num", required=True, type=str, default="1")
    args = parser.parse_args()

    print("setup dataloader and metrics")
    tracer = Tracer(carry_state=True)
    tracer.add(gain=args.gain_slow, beta=args.beta_slow)
    tracer.add(gain=args.gain_fast, beta=args.beta_fast)


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    BATCH_SIZE = 1
    EPOCHS = 10
    beta = 0.5
    bins = 100
    input_thr = 0.01
    warm_up = 20
    classifier_epochs = 200
    num_targets = 20

    path = (
        "./results/tdcl_full_online/shd/ff/ablation_2layer_450/B"
        + str(BATCH_SIZE)
        + "_bins"
        + str(bins)
        + "rotate5_beta"
        + str(beta)
        + "_"
        + str(tracer)
        + "_warmp_up"
        + str(warm_up)
        + "_inp_thr"
        + str(input_thr)
        + "_lr(8e-2_4e-3)_tmp(0.5_0.25)_gamma1.0_epochs"
        + str(EPOCHS)
        + "/"
    )

    if path[-1] != "/":
        path += "/"
    path += args.num + "/"

    fig_path = path + "plots/"
    os.makedirs(path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)
    fig, ax = tracer.gen_plot()
    fig.savefig(path + "kernel.png", bbox_inches="tight")
    info = tracer.get_info()
    with open(path + "info.json", "w") as file:
        json.dump(info, file, indent=4)

    train_loader, val_loader, test_loader = create_shd_dataloaders(
        batch_size=BATCH_SIZE, bins=bins, num_workers=4, val_split=True
    )
    x, y = next(iter(train_loader))
    print(f"data shape: {x.shape}")

    multi_metric_builder = MultiMetricBuilder()
    multi_metric_builder.add_metric("loss", lambda o: o.loss)
    multi_metric_builder.add_metric("spike_sum", lambda o: o.spikes_sum)
    multi_metric_builder.add_metric("spike_mean", lambda o: o.spikes_mean)
    multi_metric_builder.add_metric("sim_score_sum", lambda o: o.sim_score_sum)
    multi_metric_builder.add_metric("trace_mean", lambda o: o.trace_mean)
    multi_metric_builder.add_metric("update_sparcity", lambda o: o.update_sparcity)
    multi_metric_builder.add_metric("trace_sparcity", lambda o: o.trace_sparcity)

    tdcl_layer_func = partial(
        create_tdcl_block,
        beta=beta,
        spike_grad=surrogate.atan(),
        init_hidden=True,
        input_thr=input_thr,
        warm_up=warm_up,
    )
    print("setup model and trainers")
    tdcl_layer_trainer_1 = FullOnlineTDCLLayerTrainer(
        tdcl_layer=tdcl_layer_func(
            in_features=700,
            out_features=450,
            lr=8e-2,
            gamma=1.0,
            temp=0.5,
            tracer=tracer.clone(),
        ),
        train_metrics=multi_metric_builder.set_tag("train").build(),
        val_metrics=multi_metric_builder.set_tag("val").build(),
        test_metrics=multi_metric_builder.set_tag("test").build(),
    )

    train_list: list[tuple[torch.Tensor, torch.Tensor]] = []
    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)
        train_list.append((x, y))
    
    val_list: list[tuple[torch.Tensor, torch.Tensor]] = []
    for x, y in tqdm(val_loader):
        x = x.to(device)
        y = y.to(device)
        val_list.append((x, y))
    
    test_list: list[tuple[torch.Tensor, torch.Tensor]] = []
    for x, y in tqdm(test_loader):
        x = x.to(device)
        y = y.to(device)
        test_list.append((x, y))

    tdcl_trainer = FullOnlineTDCLTrainer(train_list, val_list, test_list, path)
    tdcl_trainer.add_layer_trainer(tdcl_layer_trainer_1)

    tdcl_trainer = tdcl_trainer.to(device)
    tdcl_trainer.train(
        EPOCHS,
        layer_in_pbar=0,
        layer_to_train=None,
        log_offset=EPOCHS * 0,
        train_lower_layers_too=True,
        train_tsne=False,
    )
    del train_list
    del val_list
    del test_list

    print("Few Shot Training...")
    print("classifying1")
    layers = [
        tdcl_layer_trainer_1.tdcl_layer.layer,
    ]
    train_classifier(
        layers=layers,
        features=450,
        num_classes=num_targets,
        epochs=classifier_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        key="classifying1",
        path=path,
        device=device,
    )

    print("done.")


if __name__ == "__main__":
    main()
