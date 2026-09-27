import argparse
import json
import os

import dill
import torch
from layer import create_tdcl_block
from snntorch import surrogate
from tqdm import tqdm
from util.dataloading import create_shd_dataloaders
from util.tracer import Tracer


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
    EPOCHS = 100
    beta = 0.5
    bins = 100
    input_thr = -torch.inf # no input gating
    warm_up = 20

    path = (
        "./results/tdcl_full_online/shd/ff/1_layer_450neurons_early_training_one_epoch/B"
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
        path += "1/"
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

    tdcl_layer = create_tdcl_block(
        in_features=700,
        out_features=450,
        lr=8e-2,
        gamma=torch.inf, # no dynamic gating
        temp=0.5,
        tracer=tracer.clone(),
        beta=beta,
        spike_grad=surrogate.atan(),
        init_hidden=True,
        input_thr=input_thr,
        warm_up=warm_up,
    )
    tdcl_layer.reset()
    tdcl_layer = tdcl_layer.to(device)

    tdcl_layer = tdcl_layer.train()
    spikes_list = []
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(train_loader)):
            # if i >= 1000:
            #     break
            x = x.to(device)
            y = y.to(device)

            T = x.shape[0]
            B = x.shape[1]

            input_sum = x.detach()
            input_sum = input_sum.reshape(T, B, -1)
            input_sum = input_sum.mean(dim=-1)

            x = x.detach()
            for t in range(T):
                out = tdcl_layer(x[t, ...], input_sum[t, ...], train=True)
                spikes = out[0]  # (1, N)
                spikes_list.append(spikes)

    spikes_tensor = torch.stack(spikes_list, dim=0)  # (Tall, 1, N)

    checkpoint = {}
    checkpoint[f"layer_1"] = {}
    checkpoint[f"layer_1"]["state_dict"] = tdcl_layer.state_dict()
    checkpoint[f"layer_1"]["spikes"] = spikes_tensor
    dill.dump(checkpoint, open(path + "final_checkpoint.pkl", "wb"))

    print("done.")


if __name__ == "__main__":
    main()
