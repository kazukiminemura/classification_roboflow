import argparse
import os
import pathlib
import shutil
from typing import Tuple

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO26n-cls with a Roboflow classification dataset"
    )
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY"), help="Roboflow API key")
    parser.add_argument("--workspace", default=None, help="Roboflow workspace slug")
    parser.add_argument("--project", default=None, help="Roboflow project slug")
    parser.add_argument("--version", default=None, type=int, help="Dataset version number")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Local dataset root path. If set, skip Roboflow download and use this path.",
    )
    parser.add_argument(
        "--format",
        default="folder",
        choices=["folder", "multiclass"],
        help="Dataset export format for Roboflow download",
    )
    parser.add_argument("--model", default="yolo26n-cls.pt", help="Pretrained classification model")
    parser.add_argument("--img-size", default=224, type=int, help="Input image size")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument("--epochs", default=None, type=int, help="Total epochs")
    parser.add_argument("--epochs-head", default=5, type=int, help="Compatibility option")
    parser.add_argument("--epochs-finetune", default=10, type=int, help="Compatibility option")
    parser.add_argument("--learning-rate", default=1e-3, type=float, help="Initial learning rate")
    parser.add_argument("--device", default=None, help='Device like "cpu", "0", "0,1"')
    parser.add_argument("--workers", default=8, type=int, help="Dataloader workers")
    parser.add_argument("--output-dir", default="artifacts", help="Output directory")
    parser.add_argument("--run-name", default="yolo26n_cls_finetune", help="Training run name")
    parser.add_argument("--patience", default=20, type=int, help="Early stopping patience")
    parser.add_argument(
        "--use-albumentations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable extra albumentations if package is available",
    )
    parser.add_argument(
        "--export-openvino",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export trained best.pt to OpenVINO IR automatically",
    )
    return parser.parse_args()


def download_dataset(args: argparse.Namespace) -> pathlib.Path:
    from roboflow import Roboflow

    if not args.workspace or not args.project or args.version is None:
        raise ValueError("workspace/project/version are required when --dataset-dir is not set.")
    if not args.api_key:
        raise ValueError("API key is required. Set --api-key or ROBOFLOW_API_KEY.")

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    dataset = version.download(args.format)
    return pathlib.Path(dataset.location)


def resolve_dataset_root(dataset_root: pathlib.Path, output_dir: pathlib.Path) -> Tuple[pathlib.Path, pathlib.Path]:
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    valid_dir = dataset_root / "valid"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Expected 'train' directory under: {dataset_root}. Check dataset export format."
        )

    if val_dir.exists():
        return dataset_root, train_dir

    if valid_dir.exists():
        staged_root = output_dir / "_ultralytics_dataset"
        if staged_root.exists():
            shutil.rmtree(staged_root)
        shutil.copytree(dataset_root, staged_root)
        shutil.copytree(staged_root / "valid", staged_root / "val")
        return staged_root, staged_root / "train"

    raise FileNotFoundError(
        f"Expected 'val' or 'valid' directory under: {dataset_root}. Check dataset export format."
    )


def save_labels(train_dir: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    labels_path = output_dir / "labels.txt"
    labels_path.write_text("\n".join(class_names), encoding="utf-8")
    return labels_path


def main():
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = pathlib.Path(args.dataset_dir) if args.dataset_dir else download_dataset(args)
    dataset_root, train_dir = resolve_dataset_root(dataset_root, output_dir)
    labels_path = save_labels(train_dir, output_dir)
    total_epochs = args.epochs if args.epochs is not None else (args.epochs_head + args.epochs_finetune)

    from ultralytics import YOLO

    augmentations = None
    if args.use_albumentations:
        try:
            import albumentations as A

            max_dropout = max(1, int(args.img_size * 0.2))
            augmentations = [
                A.ToGray(p=0.05),
                A.GaussNoise(p=0.1),
                A.MotionBlur(p=0.1),
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=max_dropout,
                    max_width=max_dropout,
                    fill_value=0,
                    p=0.1,
                ),
            ]
        except ImportError:
            print(
                "Warning: albumentations is not installed; grayscale/noise/blur/cutout augmentations will be skipped."
            )

    model = YOLO(args.model)
    train_kwargs = dict(
        data=str(dataset_root),
        epochs=total_epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        lr0=args.learning_rate,
        project=str(output_dir),
        name=args.run_name,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        cos_lr=True,
        degrees=90.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.2,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    if augmentations is not None:
        train_kwargs["augmentations"] = augmentations

    try:
        train_results = model.train(**train_kwargs)
    except TypeError as exc:
        if "augmentations" in train_kwargs:
            print(f"Warning: current Ultralytics version does not support `augmentations` argument: {exc}")
            train_kwargs.pop("augmentations", None)
            train_results = model.train(**train_kwargs)
        else:
            raise

    run_dir = pathlib.Path(getattr(train_results, "save_dir", output_dir / args.run_name))
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        fallback = run_dir / "weights" / "last.pt"
        if fallback.exists():
            best_weights = fallback

    if args.export_openvino:
        if best_weights.exists():
            export_model = YOLO(str(best_weights))
            openvino_output = export_model.export(format="openvino", imgsz=args.img_size)
            print(f"Exported OpenVINO IR to: {openvino_output}")
        else:
            print(f"Warning: best/last weights not found under {run_dir / 'weights'}. Skipped OpenVINO export.")

    print(f"Saved labels to: {labels_path}")
    print(f"Training outputs under: {run_dir}")


if __name__ == "__main__":
    main()
