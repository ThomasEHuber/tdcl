import json
import os
from functools import partial

import torch
from layer import create_tdcl_block
from snntorch import surrogate
from util.classifier_training import train_classifier
from util.dataloading import create_nmnist_dataloaders
from util.metric import MultiMetricBuilder
from util.tracer import Tracer

from tdcl import FullOnlineTDCLLayerTrainer, FullOnlineTDCLTrainer


def main() -> None:
    print("setup dataloader and metrics")
    tracer = Tracer(carry_state=True)
    tracer.add(gain=-0.8714, beta=0.90)
    tracer.add(gain=1.1816, beta=0.85)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    BATCH_SIZE = 1
    EPOCHS = 20
    beta = 0.5
    bins = 100
    input_thr = 1.0
    warm_up = 5
    classifier_epochs = 100
    num_targets = 10
    spike_as_grad = False

    path = (
        "./results/tdcl_full_online/nmnist/ff/neurons200" + "_spike_as_grad"
        if spike_as_grad
        else "atan_grad"
        + "/B"
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
    path += "1/"
    fig_path = path + "plots/"
    os.makedirs(path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)

    fig, ax = tracer.gen_plot()
    fig.savefig(path + "kernel.png", bbox_inches="tight")
    info = tracer.get_info()
    with open(path + "info.json", "w") as file:
        json.dump(info, file, indent=4)

    train_loader, val_loader, test_loader = create_nmnist_dataloaders(
        batch_size=BATCH_SIZE,
        bins=bins,
        filter_time=10000,
        num_workers=4,
        flatten=True,
        split_train=True,
        reset_cache=False,
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
        spike_as_grad=spike_as_grad,
    )

    print("setup model and trainers")
    tdcl_layer_trainer_1 = FullOnlineTDCLLayerTrainer(
        tdcl_layer=tdcl_layer_func(
            in_features=2312,
            out_features=200,
            lr=8e-2,
            gamma=1.0,
            temp=0.5,
            tracer=tracer.clone(),
        ),
        train_metrics=multi_metric_builder.set_tag("train").build(),
        val_metrics=multi_metric_builder.set_tag("val").build(),
        test_metrics=multi_metric_builder.set_tag("test").build(),
    )
    tdcl_layer_trainer_2 = FullOnlineTDCLLayerTrainer(
        tdcl_layer=tdcl_layer_func(
            in_features=200,
            out_features=200,
            lr=4e-3,
            gamma=1.0,
            temp=0.25,
            tracer=tracer.clone(),
        ),
        train_metrics=multi_metric_builder.set_tag("train").build(),
        val_metrics=multi_metric_builder.set_tag("val").build(),
        test_metrics=multi_metric_builder.set_tag("test").build(),
    )

    tdcl_trainer = FullOnlineTDCLTrainer(train_loader, val_loader, test_loader, path)
    tdcl_trainer.add_layer_trainer(tdcl_layer_trainer_1)
    tdcl_trainer.add_layer_trainer(tdcl_layer_trainer_2)

    tdcl_trainer = tdcl_trainer.to(device)
    tdcl_trainer.train(
        EPOCHS,
        layer_in_pbar=0,
        layer_to_train=None,
        log_offset=EPOCHS * 0,
        train_lower_layers_too=True,
        train_tsne=False,
    )

    print("Few Shot Training...")
    layers = [tdcl_layer_trainer_1.tdcl_layer.layer]
    train_classifier(
        layers=layers,
        features=200,
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
