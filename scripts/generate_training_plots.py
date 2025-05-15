#!/usr/bin/env python3
import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# 1) Point this at the directory for one run
LOGDIR = "tb_logs/controlnet3_66090/controlnet3_66090/"

# 2) Load all events
ea = event_accumulator.EventAccumulator(
    LOGDIR,
    size_guidance={ "scalars": 0 }  # 0 = load every scalar
)
ea.Reload()

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", required=True,
                    help="path to the TB run directory (the one containing events.out.tfevents.* files)")
args = parser.parse_args()

ea = EventAccumulator(args.logdir, size_guidance={"scalars":0})
ea.Reload()

print("Available scalar tags:")
for tag in ea.Tags()["scalars"]:
    print("   ", tag)

# now you can pick from that list:
# train_mse = ea.Scalars("train/epoch_mse")  # if it’s in the printed list


# 3) Grab the series you want
train_mse = ea.Scalars("train/epoch_mse")
val_mse   = ea.Scalars("val/epoch_mse")
val_psnr  = ea.Scalars("val/epoch_psnr")
val_ssim  = ea.Scalars("val/epoch_ssim")

# 4) Helper to extract step & value
def unpack(series):
    return [e.step for e in series], [e.value for e in series]

t_steps, t_vals = unpack(train_mse)
v_steps, v_vals = unpack(val_mse)
p_steps, p_vals = unpack(val_psnr)
s_steps, s_vals = unpack(val_ssim)

# 5) Plot
plt.figure(figsize=(8,6))
plt.plot(t_steps, t_vals, label="Train MSE")
plt.plot(v_steps, v_vals, label="Val   MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("MSE over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mse_curve.png")
print("Wrote mse_curve.png")

plt.figure(figsize=(8,6))
plt.plot(p_steps, p_vals, label="Val PSNR")
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.title("Validation PSNR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("psnr_curve.png")
print("Wrote psnr_curve.png")

plt.figure(figsize=(8,6))
plt.plot(s_steps, s_vals, label="Val SSIM")
plt.xlabel("Epoch")
plt.ylabel("SSIM")
plt.title("Validation SSIM")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ssim_curve.png")
print("Wrote ssim_curve.png")
