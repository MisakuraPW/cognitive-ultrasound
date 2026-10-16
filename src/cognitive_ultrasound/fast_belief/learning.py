"""Batched, supervised stages with burn-in and truncated recurrent backpropagation."""

import torch

from .model import ArrayOracle, filter_loss, rec_loss


def objective(model, clip, starts, stage, backward=False):
    """One optimizer update per batch; TBPTT windows accumulate normalized gradients."""
    cfg = model.cfg
    length = clip.shape[1]
    if stage == "codec":
        target = clip.flatten(0, 1)
        prediction = model.codec.decoder(model.codec.encoder(target))
        loss = rec_loss(target, prediction, cfg["loss"]["ssim"])
        if backward:
            loss.backward()
        return dict(total=loss.detach(), rec=loss.detach())
    if stage == "prior":
        with torch.no_grad():
            z = model.codec.encoder(clip[:, 0])
            target_z = model.codec.encoder(clip[:, 1:].flatten(0, 1)).reshape(
                len(clip), length - 1, *z.shape[1:]
            )
        terms = []
        for t in range(1, length):
            z = model.prior(z, starts + t)
            terms.append(
                (z - target_z[:, t - 1]).abs().mean()
                + rec_loss(clip[:, t], model.codec.decoder(z), cfg["loss"]["ssim"])
            )
        loss = torch.stack(terms).mean()
        if backward:
            loss.backward()
        return dict(total=loss.detach())
    training = cfg["training"]["filter"]
    burn, bptt = training["burn_in"], training["bptt"]
    if length <= burn:
        raise ValueError("No supervised filter frames after burn-in")
    with torch.no_grad():
        memory = model.initial(len(clip), clip.device)
    window, totals = [], {}
    for t in range(length):
        enabled = t >= burn
        with torch.set_grad_enabled(enabled and backward):
            memory, state = model.frame(ArrayOracle(clip[:, t]), memory, starts + t)
            if enabled:
                loss, parts = filter_loss(model, clip[:, t], state["updates"])
                window.append(loss / (length - burn))
                for key, value in dict(total=loss, **parts).items():
                    totals[key] = totals.get(key, 0) + value.detach() / (length - burn)
        boundary = enabled and (len(window) == bptt or t == length - 1)
        if boundary:
            if backward:
                torch.stack(window).sum().backward()
            window.clear()
            memory = memory.detach()
        elif not enabled:
            memory = memory.detach()
    return totals


def update(model, optimizer, clip, starts, stage):
    optimizer.zero_grad(set_to_none=True)
    terms = objective(model, clip, starts, stage, backward=True)
    if not torch.stack([torch.isfinite(v).all() for v in terms.values()]).all():
        raise FloatingPointError("Nonfinite training loss")
    params = [p for p in model.parameters() if p.requires_grad]
    if (
        any(p.grad is None for p in params)
        or not torch.stack([torch.isfinite(p.grad).all() for p in params]).all()
    ):
        raise FloatingPointError("Absent/nonfinite parameter gradient")
    norm = torch.nn.utils.clip_grad_norm_(
        params, model.cfg["training"]["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    if not torch.stack([torch.isfinite(p).all() for p in params]).all():
        raise FloatingPointError("Nonfinite updated parameter")
    return {**{k: float(v) for k, v in terms.items()}, "gradient_norm": float(norm)}
