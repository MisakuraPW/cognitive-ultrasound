import argparse
import json

from .config import ROOT, load, path


def parser():
    p = argparse.ArgumentParser(description="CASL reproduction; run commands from any directory")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("doctor", help="Record actual local environment")
    q.add_argument("--output", default="reports/environment.md")
    q = sub.add_parser(
        "fetch-assets", help="Download pinned official weights and patient split lists"
    )
    q.add_argument("--checkpoint", default="checkpoints/official")
    q.add_argument("--splits", default="configs/splits")
    q.add_argument("--splits-only", action="store_true")
    q.add_argument(
        "--with-evaluation",
        action="store_true",
        help="Also fetch pinned LPIPS and segmentation weights",
    )
    q = sub.add_parser("prepare-data", help="Official EchoNet AVI -> polar HDF5 conversion")
    q.add_argument("--raw", required=True, help="EchoNet-Dynamic root containing Videos")
    q.add_argument("--output", default="data/echonet-polar")
    q.add_argument("--manifest", default="configs/splits/split.yaml")
    q.add_argument("--workers", type=int, default=1, help="Conversion processes (1-32)")
    q.add_argument("--resume", action="store_true", help="Validate and reuse complete conversions")
    q = sub.add_parser("audit-data")
    q.add_argument("--data-root", default="data/echonet-polar")
    q.add_argument("--manifest", default="configs/splits/split.yaml")
    q.add_argument("--output", default="reports/dataset_statistics.md")
    q.add_argument("--allow-partial", action="store_true")
    q = sub.add_parser("evaluate")
    q.add_argument("--config", default="configs/baseline.yaml")
    q.add_argument("--data-root")
    q.add_argument("--checkpoint")
    q.add_argument("--evaluation-checkpoints")
    q.add_argument("--output")
    q.add_argument("--methods", nargs="+", choices=["random", "uniform", "casl"])
    q.add_argument("--budgets", nargs="+", type=int)
    q.add_argument("--metrics", nargs="+", choices=["psnr", "ssim", "lpips"])
    q.add_argument("--split", choices=["val", "test"])
    q.add_argument("--frames", type=int)
    q.add_argument("--limit-cases", type=int)
    q.add_argument("--profile", action="store_true", default=None)
    q.add_argument("--segmentation", action="store_true", default=None)
    q.add_argument("--resume", action="store_true")
    q = sub.add_parser("train")
    q.add_argument("--config", default="configs/training.yaml")
    q.add_argument("--output")
    q.add_argument("--precision", choices=["float32", "mixed_float16"])
    q.add_argument("--resume", action="store_true")
    q.add_argument(
        "--smoke", action="store_true", help="Synthetic one-step training, not research evidence"
    )
    q = sub.add_parser("report")
    q.add_argument("run")
    q = sub.add_parser("smoke", help="CPU synthetic artifact/metric check; does NOT execute CASL")
    q.add_argument("--output", default="results/synthetic_smoke")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "doctor":
        from .provenance import environment_report

        environment_report(path(args.output))
        print(path(args.output))
    elif args.command == "fetch-assets":
        from .data import fetch_assets

        fetch_assets(path(args.checkpoint), path(args.splits), not args.splits_only)
        if args.with_evaluation:
            from .assets import fetch_evaluation_assets

            fetch_evaluation_assets(ROOT / "checkpoints/evaluation")
    elif args.command == "prepare-data":
        from .data import audit_dataset, convert

        convert(
            path(args.raw),
            path(args.output),
            path(args.manifest),
            args.workers,
            args.resume,
        )
        audit_dataset(
            path(args.output),
            path(args.manifest),
            ROOT / "reports/dataset_statistics.md",
        )
    elif args.command == "audit-data":
        from .data import audit_dataset

        audit_dataset(
            path(args.data_root), path(args.manifest), path(args.output), not args.allow_partial
        )
    elif args.command == "evaluate":
        from .experiments import evaluate

        cfg = load(path(args.config))
        for key, value in vars(args).items():
            if key in cfg and value is not None:
                cfg[key] = value
        evaluate(cfg, resume=args.resume)
    elif args.command == "train":
        from .training import train

        cfg = load(path(args.config))
        for key in ("output", "precision"):
            if getattr(args, key) is not None:
                cfg[key] = getattr(args, key)
        if args.smoke and args.output is None:
            cfg["output"] = "checkpoints/synthetic_training_smoke"
        train(cfg, resume=args.resume, smoke=args.smoke)
    elif args.command == "report":
        from .evaluation.report import generate_report

        print(generate_report(path(args.run)))
    elif args.command == "smoke":
        from .experiments import synthetic_smoke

        print(json.dumps(synthetic_smoke(path(args.output)), indent=2))


if __name__ == "__main__":
    main()
