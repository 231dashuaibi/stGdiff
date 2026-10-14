import os

import torch
import torch.nn as nn
from tqdm import tqdm
from ray.air import session

import matplotlib
matplotlib.use('Agg')

from .stGdiff_scheduler import NoiseScheduler


def normal_train_stGdiff(model,
                 dataloader,
                 lr: float = 1e-4,
                 num_epoch: int = 1400,
                 pred_type: str = 'noise',
                 diffusion_step: int = 1000,
                 device=torch.device('cpu'),
                 is_tqdm: bool = True,
                 is_tune: bool = False,
                 mask = None,
                 loss_save_path: str = None
                       ):
    if mask is not None:
        mask = torch.tensor(mask, dtype=torch.float32, device=device)
        mask = mask[None, :]
        no_mask = 1.0 - mask
    else:
        mask = torch.tensor(0.0, dtype=torch.float32, device=device)
        no_mask = torch.tensor(1.0, dtype=torch.float32, device=device)

    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )
    criterion = nn.HuberLoss(delta=1)
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6, betas=(0.9, 0.999))

    if is_tqdm:
        t_epoch = tqdm(range(num_epoch), ncols=100)
    else:
        t_epoch = range(num_epoch)

    model.train()
    loss_history = []
    for epoch in t_epoch:
        epoch_loss = 0.
        for i, (x, x_cond, LC) in enumerate(dataloader):
            x = x.to(device)
            x_cond = x_cond.to(device)
            LC = LC.to(device)

            noise = torch.randn_like(x, device=device)
            timesteps = torch.randint(
                1, diffusion_step, (x.shape[0],),
                dtype=torch.long, device=device
            )
            x_t = noise_scheduler.add_noise(x, noise, timesteps=timesteps)

            x_noisy = x_t * no_mask + x * mask

            with torch.cuda.amp.autocast(enabled=use_amp):
                noise_pred = model(x_noisy, t=timesteps, y=x_cond, LC=LC)
                loss = criterion(noise * no_mask, noise_pred * no_mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / (i + 1)
        if is_tqdm:
            t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.5f}')
        if is_tune:
            session.report({'loss': epoch_loss})
