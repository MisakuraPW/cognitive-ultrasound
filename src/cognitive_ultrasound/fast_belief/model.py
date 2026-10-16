"""Paper4.3--4.19, with explicit architecture/selection assumptions in the config."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        groups = next(g for g in (8, 4, 2, 1) if b % g == 0)
        self.body = nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1),
            nn.GroupNorm(groups, b),
            nn.SiLU(),
            nn.Conv2d(b, b, 3, padding=1),
            nn.GroupNorm(groups, b),
        )
        self.skip = nn.Identity() if a == b else nn.Conv2d(a, b, 1)

    def forward(self, x):
        return F.silu(self.skip(x) + self.body(x))


class LinearResize(nn.Module):
    """Fixed square bilinear align_corners=False via deterministic linear maps.

    CUDA interpolate's bilinear backward has no deterministic implementation.
    The two separable maps retain bilinear values without its atomic scatter backward.
    """

    def __init__(self, source, target):
        super().__init__()
        position = (
            (torch.arange(target, dtype=torch.float64) + 0.5) * source / target - 0.5
        ).clamp(0, source - 1)
        left = position.floor().long()
        right = (left + 1).clamp_max(source - 1)
        weight = position - left
        matrix = torch.zeros(target, source, dtype=torch.float64)
        matrix.scatter_add_(1, left[:, None], (1 - weight)[:, None])
        matrix.scatter_add_(1, right[:, None], weight[:, None])
        self.register_buffer("matrix", matrix.float(), persistent=False)

    def forward(self, x):
        x = F.linear(x, self.matrix)
        return F.linear(x.transpose(-1, -2), self.matrix).transpose(-1, -2)


class Codec(nn.Module):
    def __init__(self, width, latent):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, width, 5, 2, 2),
            nn.SiLU(),
            Residual(width, width),
            nn.Conv2d(width, width * 2, 3, 2, 1),
            Residual(width * 2, width * 2),
            nn.Conv2d(width * 2, latent, 1),
        )
        self.decoder = nn.Sequential(
            Residual(latent, width * 2),
            LinearResize(28, 56),
            Residual(width * 2, width),
            LinearResize(56, 112),
            Residual(width, width),
            nn.Conv2d(width, 1, 3, padding=1),
            nn.Tanh(),
        )


class Prior(nn.Module):
    def __init__(self, width, latent, limit):
        super().__init__()
        self.body = nn.Sequential(Residual(latent + 1, width), Residual(width, width))
        self.head = nn.Conv2d(width, latent, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.limit = limit

    def forward(self, z, time):
        time = torch.as_tensor(time, dtype=z.dtype, device=z.device).reshape(-1, 1, 1, 1)
        time = time.expand(z.shape[0], 1, z.shape[2], z.shape[3])
        return z + self.limit * self.head(self.body(torch.cat([z, time / (time + 100)], 1))).tanh()


class Updater(nn.Module):
    def __init__(self, width, latent, limit):
        super().__init__()
        self.stem = Residual(latent * 3 + 4, width)
        self.down1 = nn.Sequential(
            nn.Conv2d(width, 2 * width, 3, 2, 1), Residual(2 * width, 2 * width)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(2 * width, 4 * width, 3, 2, 1), Residual(4 * width, 4 * width)
        )
        self.up1 = Residual(6 * width, 2 * width)
        self.up2 = Residual(3 * width, width)
        self.delta = nn.Conv2d(width, latent, 1)
        self.gate = nn.Conv2d(width, latent, 1)
        self.logvar = nn.Conv2d(width, 1, 1)
        self.resize1 = LinearResize(7, 14)
        self.resize2 = LinearResize(14, 28)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 1.0)
        self.limit = limit

    def forward(self, features):
        a = self.stem(features)
        b = self.down1(a)
        c = self.down2(b)
        x = self.up1(torch.cat([self.resize1(c), b], 1))
        x = self.up2(torch.cat([self.resize2(x), a], 1))
        return (
            self.limit * self.delta(x).tanh(),
            self.gate(x).sigmoid(),
            self.logvar(x).clamp(-6, 4),
        )


def project(raw, observation, mask):
    return torch.where(mask.bool(), observation, raw)


def ssim(x, y):
    """Local Gaussian11/1.5 SSIM on continuous[-1,1], data_range2; per image."""
    coords = torch.arange(11, dtype=x.dtype, device=x.device) - 5
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :])[None, None]
    ux, uy = F.conv2d(x, kernel), F.conv2d(y, kernel)
    vx = (F.conv2d(x * x, kernel) - ux.square()).clamp_min(0)
    vy = (F.conv2d(y * y, kernel) - uy.square()).clamp_min(0)
    cov = F.conv2d(x * y, kernel) - ux * uy
    return (
        ((2 * ux * uy + 0.02**2) * (2 * cov + 0.06**2))
        / ((ux.square() + uy.square() + 0.02**2) * (vx + vy + 0.06**2))
    ).mean((1, 2, 3))


def rec_loss(x, y, weight):
    return (x - y).abs().mean() + weight * (1 - ssim(x, y).mean())


def normalize(x):
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    return torch.where(hi - lo > 1e-8, (x - lo) / (hi - lo).clamp_min(1e-8), torch.zeros_like(x))


def smooth(x, sigma):
    radius = int(3 * sigma)
    k = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    g = torch.exp(-0.5 * (k / sigma) ** 2)
    g = (g / g.sum())[None, None]
    return F.conv1d(F.pad(x[:, None], (radius, radius), mode="replicate"), g)[:, 0]


@torch.no_grad()
def select(scores, used, count, penalty, sigma):
    """Batch-vectorized greedy; deterministic exact ties; no CPU synchronization."""
    if scores.ndim != 2 or scores.shape[1] != 112 or used.shape != scores.shape:
        raise ValueError("Expected Bx112 scores and acquired lines")
    if count < 0 or count > 112:
        raise ValueError("Invalid acquisition count")
    excluded = used.clone()
    positions = torch.arange(112, dtype=scores.dtype, device=scores.device)
    costs = torch.zeros_like(scores)
    # Exact ties: closest to centre, then left. Does not perturb unequal scores.
    priority = -(positions - 55.5).abs() * 112 - positions / 112
    choices = []
    for _ in range(count):
        value = (scores - penalty * costs).masked_fill(excluded, -torch.inf)
        ties = value == value.amax(1, keepdim=True)
        line = priority.expand_as(value).masked_fill(~ties, -torch.inf).argmax(1)
        choices.append(line)
        excluded.scatter_(1, line[:, None], True)
        costs += torch.exp(-0.5 * ((positions[None] - line[:, None]) / sigma) ** 2)
    return (
        torch.stack(choices, 1)
        if choices
        else torch.empty((len(scores), 0), dtype=torch.long, device=scores.device)
    )


@dataclass
class Memory:
    z: torch.Tensor
    image: torch.Tensor
    uncertainty: torch.Tensor
    values: torch.Tensor

    def detach(self):
        return Memory(*(x.detach() for x in (self.z, self.image, self.uncertainty, self.values)))


class BeliefSystem(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        m = cfg["model"]
        self.codec = Codec(m["width"], m["latent_channels"])
        self.prior = Prior(m["width"], m["latent_channels"], m["delta_limit"])
        self.updater = Updater(m["width"], m["latent_channels"], m["delta_limit"])
        self.image_resize = LinearResize(28, 112)

    def stage(self, name):
        self.train()
        active = {"codec": self.codec, "prior": self.prior, "filter": self.updater}[name]
        for p in self.parameters():
            p.requires_grad_(False)
        for p in active.parameters():
            p.requires_grad_(True)
        return list(active.parameters())

    def initial(self, batch, device):
        z = torch.zeros(batch, self.cfg["model"]["latent_channels"], 28, 28, device=device)
        return Memory(
            z,
            self.codec.decoder(z),
            torch.ones(batch, 1, 112, 112, device=device),
            torch.zeros(batch, 112, device=device),
        )

    def update(self, z, observation, mask, before=None, history=None):
        if before is None:
            before = self.codec.decoder(z)
        projected = project(before, observation, mask)
        zdc = self.codec.encoder(projected)
        error = mask * (observation - before)
        # Separate previous groups' within-frame history from cumulative observations.
        history = torch.zeros_like(mask) if history is None else history
        features = torch.cat(
            [z, zdc, zdc - z, *[F.avg_pool2d(x, 4) for x in (mask, error, error.abs(), history)]], 1
        )
        delta, gate, logvar = self.updater(features)
        new = z + gate * (zdc - z) + delta
        raw = self.codec.decoder(new)
        uncertainty = self.image_resize(F.softplus(logvar))
        return new, raw, project(raw, observation, mask), uncertainty, logvar, projected

    def frame(self, oracle, memory, time, policy="greedy"):
        """Oracle only reveals requested columns. Decisions receive no full target."""
        batch = memory.z.shape[0]
        cfg = self.cfg["selection"]
        z = self.prior(memory.z, time)
        image = self.codec.decoder(z)
        before = image
        change = (image - memory.image).detach().abs().mean((1, 2))
        uncertainty = memory.uncertainty
        values = memory.values.clone()
        used = torch.zeros(batch, 112, dtype=torch.bool, device=z.device)
        mask = torch.zeros(batch, 1, 112, 112, device=z.device)
        observed = torch.zeros_like(mask)
        error = torch.zeros_like(mask)
        updates, actions = [], []
        offset = 0
        for group, count in enumerate(cfg["groups"]):
            history = mask
            with torch.no_grad():
                w = cfg["weights"]
                scores = (
                    w["uncertainty"] * normalize(uncertainty.detach().mean((1, 2)))
                    + w["change"] * normalize(change)
                    + w["value"] * normalize(values)
                )
                if group:
                    scores += w["residual"] * normalize(
                        smooth(error.detach().mean((1, 2)), cfg["residual_sigma"])
                    )
                    scores -= w["history"] * normalize(
                        smooth(mask.mean((1, 2)), cfg["redundancy_sigma"])
                    )
                if policy == "greedy":
                    lines = select(scores, used, count, cfg["redundancy"], cfg["redundancy_sigma"])
                elif policy == "uniform":
                    fixed = (
                        torch.linspace(0, 111, sum(cfg["groups"]), device=z.device).round().long()
                    )
                    lines = fixed[offset : offset + count].expand(batch, -1)
                else:
                    raise ValueError("Unknown policy")
                used.scatter_(1, lines, True)
                newmask = (
                    torch.zeros_like(used).scatter_(1, lines, True)[:, None, None].expand_as(mask)
                )
                measured = oracle.acquire(lines)
                observed = torch.where(newmask, measured, observed)
                mask = used[:, None, None].expand_as(mask).to(z.dtype)
                error = mask * (observed - before.detach()).abs()
                measured_error = error.mean((1, 2))
                values = torch.where(
                    newmask[:, 0, 0],
                    cfg["value_decay"] * values + (1 - cfg["value_decay"]) * measured_error,
                    values,
                )
            z, raw, image, uncertainty, logvar, projected = self.update(
                z, observed, mask, before, history
            )
            before = raw
            updates.append((z, raw, logvar, projected, observed, mask))
            actions.append(lines)
            offset += count
        return Memory(z, image, uncertainty, values.detach()), dict(
            image=image,
            mask=mask,
            observed=observed,
            uncertainty=uncertainty,
            actions=actions,
            updates=updates,
        )


class ArrayOracle:
    def __init__(self, target):
        self.target = target

    def acquire(self, lines):
        mask = torch.zeros(
            (len(self.target), 112), dtype=torch.bool, device=self.target.device
        ).scatter_(1, lines, True)
        return torch.where(mask[:, None, None], self.target, torch.zeros_like(self.target))


def filter_loss(model, target, updates):
    weights = model.cfg["loss"]
    with torch.no_grad():
        target_z = model.codec.encoder(target)
    terms = {k: [] for k in ("rec", "obs", "lat", "unc", "proj")}
    for z, raw, logvar, projected, observed, mask in updates:
        terms["rec"].append(rec_loss(target, raw, weights["ssim"]))
        terms["obs"].append(((raw - observed).abs() * mask).sum() / mask.sum().clamp_min(1))
        terms["lat"].append((z - target_z).abs().mean())
        logscale = model.image_resize(logvar)
        terms["unc"].append(((-logscale).exp() * (raw - target).square() + logscale).mean())
        terms["proj"].append((raw - projected).abs().mean())
    terms = {k: torch.stack(v).mean() for k, v in terms.items()}
    return sum(weights[k] * value for k, value in terms.items()), terms
