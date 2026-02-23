import argparse
import os
import pathlib
from typing import Tuple

import tensorflow as tf
from dotenv import load_dotenv
from roboflow import Roboflow


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Train/fine-tune MobileNetV2 with a Roboflow classification dataset"
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
    parser.add_argument("--img-size", default=224, type=int, help="Input image size")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument("--epochs-head", default=5, type=int, help="Epochs for classifier head training")
    parser.add_argument("--epochs-finetune", default=10, type=int, help="Epochs for fine-tuning")
    parser.add_argument(
        "--finetune-layers",
        default=30,
        type=int,
        help="Number of MobileNetV2 top layers to unfreeze during fine-tuning",
    )
    parser.add_argument("--learning-rate-head", default=1e-3, type=float, help="LR for head training")
    parser.add_argument("--learning-rate-finetune", default=1e-5, type=float, help="LR for fine-tuning")
    parser.add_argument("--output-dir", default="artifacts", help="Output directory")
    return parser.parse_args()


def download_dataset(args: argparse.Namespace) -> pathlib.Path:
    if not args.workspace or not args.project or args.version is None:
        raise ValueError(
            "workspace/project/version are required when --dataset-dir is not set."
        )
    if not args.api_key:
        raise ValueError("API key is required. Set --api-key or ROBOFLOW_API_KEY.")

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    dataset = version.download(args.format)
    return pathlib.Path(dataset.location)


def resolve_data_dirs(dataset_root: pathlib.Path) -> Tuple[pathlib.Path, pathlib.Path]:
    train_dir = dataset_root / "train"
    valid_dir = dataset_root / "valid"

    if not train_dir.exists() or not valid_dir.exists():
        raise FileNotFoundError(
            f"Expected train/valid directories under: {dataset_root}. "
            "Check Roboflow export format (classification folder)"
        )
    return train_dir, valid_dir


def make_datasets(
    train_dir: pathlib.Path,
    valid_dir: pathlib.Path,
    img_size: int,
    batch_size: int,
):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    valid_ds = valid_ds.prefetch(autotune)
    return train_ds, valid_ds, class_names


def build_model(num_classes: int, img_size: int) -> Tuple[tf.keras.Model, tf.keras.Model]:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def main():
    args = parse_args()
    dataset_root = pathlib.Path(args.dataset_dir) if args.dataset_dir else download_dataset(args)
    train_dir, valid_dir = resolve_data_dirs(dataset_root)
    train_ds, valid_ds, class_names = make_datasets(
        train_dir, valid_dir, args.img_size, args.batch_size
    )

    model, base_model = build_model(len(class_names), args.img_size)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ]

    compile_model(model, args.learning_rate_head)
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs_head,
        callbacks=callbacks,
    )

    # Fine-tuning: unfreeze last N layers of base model except BatchNorm.
    base_model.trainable = True
    if args.finetune_layers > 0:
        for layer in base_model.layers[:-args.finetune_layers]:
            layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, args.learning_rate_finetune)
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs_head + args.epochs_finetune,
        initial_epoch=args.epochs_head,
        callbacks=callbacks,
    )

    final_model_path = output_dir / "mobilenetv2_finetuned.keras"
    model.save(final_model_path)

    labels_path = output_dir / "labels.txt"
    labels_path.write_text("\n".join(class_names), encoding="utf-8")

    print(f"Saved best model to: {checkpoint_path}")
    print(f"Saved final model to: {final_model_path}")
    print(f"Saved labels to: {labels_path}")


if __name__ == "__main__":
    main()
