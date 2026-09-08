from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CASL_COMMIT = "5f57aba668eb34054b8ced817c666b7d18a51ca2"
ZEA_COMMIT = "192c0bbd4e89061c38048048673beadcb8723a82"
MODEL_REVISION = "74322bb9e2bd01c6f994cd5435527e66792d2866"
SPLIT_REVISION = "534aa314fe5a54912483b7cab936ec023f934d10"
LPIPS_REVISION = "86cc9154cf1e1af4c4e27fd123fd8c95b85383c7"
SEGMENTATION_REVISION = "b594794cde1afb9741c34436856af41549054c82"


def path(value):
    value = Path(value).expanduser()
    return value if value.is_absolute() else ROOT / value


def load(file, _parents=()):
    file = Path(file).resolve()
    if file in _parents:
        raise ValueError(f"Configuration inheritance cycle: {file}")
    with open(file, encoding="utf-8") as stream:
        result = yaml.safe_load(stream)
    if not isinstance(result, dict):
        raise ValueError("Configuration must be a YAML mapping")
    parent = result.pop("extends", None)
    if parent is not None:
        if not isinstance(parent, str):
            raise ValueError("extends must be a YAML filename")
        base = load(file.parent / parent, (*_parents, file))
        result = _merge(base, result)
    return result


def _merge(base, override):
    result = dict(base)
    for key, value in override.items():
        result[key] = (
            _merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def validate(cfg):
    if cfg["backend"] != "jax":
        raise ValueError("The official CASL inference adapter requires JAX")
    if cfg["split"] not in ("val", "test"):
        raise ValueError("Evaluation must use val or test")
    if cfg["precision"] != "float32":
        raise ValueError("Quality baseline uses float32; speed experiments need separate configs")
    if cfg["temporal_window"] != 3:
        raise ValueError("Paper baseline requires W=3")
    if cfg["particles"] < 2:
        raise ValueError("Entropy requires at least two posterior particles")
    if not 0 <= cfg["initial_step"] < cfg["num_steps"]:
        raise ValueError("Expected 0 <= initial_step < num_steps")
    if cfg["frames"] < 1 or cfg.get("limit_cases") is not None and cfg["limit_cases"] < 1:
        raise ValueError("frames and limit_cases must be positive")
    if not cfg["budgets"] or any(type(k) is not int or not 1 <= k <= 112 for k in cfg["budgets"]):
        raise ValueError("Budgets must be integers in [1,112]")
    if len(set(cfg["budgets"])) != len(cfg["budgets"]):
        raise ValueError("Duplicate budgets")
    if not cfg["methods"] or set(cfg["methods"]) - {"random", "uniform", "casl"}:
        raise ValueError("Unknown or empty methods")
    if len(set(cfg["methods"])) != len(cfg["methods"]):
        raise ValueError("Duplicate methods")
    if cfg["reconstruction"] not in ("choose_first", "mean"):
        raise ValueError("Unknown reconstruction")
    if cfg["metric_domain"] != "polar_uint8":
        raise ValueError("Only the audited upstream polar uint8 metric domain is supported")
    if set(cfg["metrics"]) - {"psnr", "ssim", "lpips"} or not cfg["metrics"]:
        raise ValueError("Unknown or empty metrics")
    if cfg["image_range"] != [-60, 0]:
        raise ValueError("EchoNet polar preprocessing uses [-60,0]")
    return cfg
