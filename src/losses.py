import torch.nn as nn

adversarial_loss = nn.BCEWithLogitsLoss()
l1_loss          = nn.L1Loss()
