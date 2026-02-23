import argparse
import os
import pathlib
import shutil
from typing import Tuple

from dotenv import load_dotenv
from roboflow import Roboflow


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
    return parser.parse_args()


def download_dataset(args: argparse.Namespace) -> pathlib.Path:
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

    model = YOLO(args.model)
    model.train(
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
    )

    print(f"Saved labels to: {labels_path}")
    print(f"Training outputs under: {output_dir / args.run_name}")


if __name__ == "__main__":
    main()
