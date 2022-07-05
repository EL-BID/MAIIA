import os
import tempfile
from glob import glob
from datetime import datetime

import click
import satproc.chips
import unetseg.train


def extract_chips(
    image_paths,
    output_path,
    aoi_path,
    labels_path,
    size=100,
    step_size=50,
    num_channels=4,
    rescale_mode="percentiles",
    lower_cut=2,
    upper_cut=98
):
    satproc.chips.extract_chips(
        rasters=image_paths,
        output_dir=output_path,
        size=size,
        step_size=step_size,
        aoi=aoi_path,
        labels=labels_path,
        label_property="class",
        bands=range(1, num_channels + 1),
        classes=["A"],
        rescale_mode=rescale_mode,
        rescale_range=(lower_cut, upper_cut),
    )


def train(
    images_path,
    model_path,
    num_channels=4,
    unet_size=160,
    epochs=30,
    steps_per_epoch=100,
    batch_size=16,
    seed=42
):
    config = unetseg.train.TrainConfig(
        width=unet_size,
        height=unet_size,
        n_channels=num_channels,
        n_classes=1,
        apply_image_augmentation=True,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        early_stopping_patience=10,
        validation_split=0.1,
        test_split=0.1,
        model_architecture="unet",
        images_path=images_path,
        model_path=model_path,
        evaluate=True,
        class_weights=[1],
    )

    return unetseg.train.train(config)


@click.command(context_settings={'show_default': True})
@click.option('--images-dir', help='Path to directory containing images')
@click.option('--output-model-path', help='Path to output model (.h5)')
@click.option('--labels-path', help='Path to labels vector file')
@click.option('--aoi-path', help='Path to AOI vector file')
@click.option("--size", default=600, help="Size of the extracted chips")
@click.option("--step-size", default=100, help="Step size of the extracted chips")
@click.option("--num-channels", default=3, help="Number of channels of input images")
@click.option("--epochs", "-E", default=30, help="Number of epochs to train model")
@click.option("--steps-per-epoch", "-s", default=100, help="Number of steps per epoch")
@click.option("--batch-size", "-B", default=16, help="Batch size")
@click.option("--seed", "-S", default=42, help="Random seed")
@click.option("--temp-dir", help="Path to temporary directory, which will contain extracted chips")
def main(images_dir, output_model_path, labels_path, aoi_path, size, step_size, num_channels, epochs, steps_per_epoch, batch_size, seed, temp_dir):
    dt_s = datetime.today().strftime("%Y%m%d_%H%M%S")

    if not images_dir:
        images_dir = os.path.join("data", "train", "images")

    image_paths = glob(os.path.join(images_dir, "**", '*.tif'), recursive=True)
    if not image_paths:
        raise click.UsageError(f"No .tif images found in {images_dir}")
    click.echo(f"Found {len(image_paths)} images on {images_dir}")

    if not output_model_path:
        output_model_path = os.path.join("data", "models", f"model_{dt_s}.h5")

    if not labels_path:
        # Try to look for labels files on data/train/labels directory
        labels_dir = os.path.join("data", "train", "labels")
        label_files = glob(os.path.join(labels_dir, "*.gpkg"))
        label_files = sorted(label_files, key=lambda t: -os.stat(t).st_mtime)
        if not label_files:
            raise click.UsageError(f"No .gpkg files found in {labels_dir}")
        labels_path = label_files[0]
    click.echo(f"Using {labels_path} as labels file")

    if not aoi_path:
        # Try to look for AOI files on data/train/areas directory
        areas_dir = os.path.join("data", "train", "areas")
        areas_files = glob(os.path.join(areas_dir, "*.gpkg"))
        areas_files = sorted(areas_files, key=lambda t: -os.stat(t).st_mtime)
        if not areas_files:
            aoi_path = labels_path
            click.echo(f"No custom AOI path was specified and no files were found in {areas_dir}. Will use the same labels vector file by default.")
        else:
            aoi_path = areas_files[0]
    click.echo(f"Using {aoi_path} as AOI file")

    if not temp_dir:
        temp_dir = os.path.join("data", "tmp")

    chips_path = os.path.join(temp_dir, "chips", f"{size}_{step_size}")
    click.echo(f"Extracting chips of size {size} with step size {step_size} into {chips_path}")

    extract_chips(
        image_paths,
        chips_path,
        aoi_path,
        labels_path,
        size=size,
        step_size=step_size,
        num_channels=num_channels,
        rescale_mode="percentiles",
        lower_cut=2,
        upper_cut=98
    )

    train(
        chips_path,
        output_model_path,
        unet_size=160,
        num_channels=num_channels,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        seed=seed
    )

    click.echo(f"Training finished! Model saved to {output_model_path}")


if __name__ == "__main__":
    main()
