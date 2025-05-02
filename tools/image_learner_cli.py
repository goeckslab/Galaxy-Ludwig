#!/usr/bin/env python3
import argparse
import logging
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import zipfile
import json
import sys
import os
# import subprocess # No longer needed for the backend
import yaml
from sklearn.model_selection import train_test_split
import traceback
from typing import Protocol, Dict, Any, Tuple, Optional, List
from utils import encode_image_to_base64, get_html_closing, get_html_template
from ludwig.visualize import get_visualizations_registry
from ludwig.globals import (
    DESCRIPTION_FILE_NAME,
    PREDICTIONS_PARQUET_FILE_NAME,
    TEST_STATISTICS_FILE_NAME,
    TRAIN_SET_METADATA_FILE_NAME,
)
from ludwig.utils.data_utils import get_split_path

# --- Constants ---
SPLIT_COLUMN_NAME = 'split'
LABEL_COLUMN_NAME = 'label'
IMAGE_PATH_COLUMN_NAME = 'image_path'
DEFAULT_SPLIT_PROBABILITIES = [0.7, 0.1, 0.2]
TEMP_CSV_FILENAME = "processed_data_for_ludwig.csv"
TEMP_CONFIG_FILENAME = "ludwig_config.yaml"
TEMP_DIR_PREFIX = "ludwig_api_work_" # Changed prefix slightly
MODEL_ENCODER_TEMPLATES: Dict[str, Any] = {
    'stacked_cnn': 'stacked_cnn',
    'resnet18': {'type': 'resnet', 'model_variant': 18},
    'resnet34': {'type': 'resnet', 'model_variant': 34},
    'resnet50': {'type': 'resnet', 'model_variant': 50},
    'resnet101': {'type': 'resnet', 'model_variant': 101},
    'resnet152': {'type': 'resnet', 'model_variant': 152},
    'resnext50_32x4d': {'type': 'resnext', 'model_variant': '50_32x4d'},
    'resnext101_32x8d': {'type': 'resnext', 'model_variant': '101_32x8d'},
    'resnext101_64x4d': {'type': 'resnext', 'model_variant': '101_64x4d'},
    'resnext152_32x8d': {'type': 'resnext', 'model_variant': '152_32x8d'},
    'wide_resnet50_2': {'type': 'wide_resnet', 'model_variant': '50_2'},
    'wide_resnet101_2': {'type': 'wide_resnet', 'model_variant': '101_2'},
    'wide_resnet103_2': {'type': 'wide_resnet', 'model_variant': '103_2'},
    'efficientnet_b0': {'type': 'efficientnet', 'model_variant': 'b0'},
    'efficientnet_b1': {'type': 'efficientnet', 'model_variant': 'b1'},
    'efficientnet_b2': {'type': 'efficientnet', 'model_variant': 'b2'},
    'efficientnet_b3': {'type': 'efficientnet', 'model_variant': 'b3'},
    'efficientnet_b4': {'type': 'efficientnet', 'model_variant': 'b4'},
    'efficientnet_b5': {'type': 'efficientnet', 'model_variant': 'b5'},
    'efficientnet_b6': {'type': 'efficientnet', 'model_variant': 'b6'},
    'efficientnet_b7': {'type': 'efficientnet', 'model_variant': 'b7'},
    'efficientnet_v2_s': {'type': 'efficientnet', 'model_variant': 'v2_s'},
    'efficientnet_v2_m': {'type': 'efficientnet', 'model_variant': 'v2_m'},
    'efficientnet_v2_l': {'type': 'efficientnet', 'model_variant': 'v2_l'},
    'regnet_y_400mf': {'type': 'regnet', 'model_variant': 'y_400mf'},
    'regnet_y_800mf': {'type': 'regnet', 'model_variant': 'y_800mf'},
    'regnet_y_1_6gf': {'type': 'regnet', 'model_variant': 'y_1_6gf'},
    'regnet_y_3_2gf': {'type': 'regnet', 'model_variant': 'y_3_2gf'},
    'regnet_y_8gf': {'type': 'regnet', 'model_variant': 'y_8gf'},
    'regnet_y_16gf': {'type': 'regnet', 'model_variant': 'y_16gf'},
    'regnet_y_32gf': {'type': 'regnet', 'model_variant': 'y_32gf'},
    'regnet_y_128gf': {'type': 'regnet', 'model_variant': 'y_128gf'},
    'regnet_x_400mf': {'type': 'regnet', 'model_variant': 'x_400mf'},
    'regnet_x_800mf': {'type': 'regnet', 'model_variant': 'x_800mf'},
    'regnet_x_1_6gf': {'type': 'regnet', 'model_variant': 'x_1_6gf'},
    'regnet_x_3_2gf': {'type': 'regnet', 'model_variant': 'x_3_2gf'},
    'regnet_x_8gf': {'type': 'regnet', 'model_variant': 'x_8gf'},
    'regnet_x_16gf': {'type': 'regnet', 'model_variant': 'x_16gf'},
    'regnet_x_32gf': {'type': 'regnet', 'model_variant': 'x_32gf'},
    'vgg11': {'type': 'vgg', 'model_variant': 11},
    'vgg11_bn': {'type': 'vgg', 'model_variant': '11_bn'},
    'vgg13': {'type': 'vgg', 'model_variant': 13},
    'vgg13_bn': {'type': 'vgg', 'model_variant': '13_bn'},
    'vgg16': {'type': 'vgg', 'model_variant': 16},
    'vgg16_bn': {'type': 'vgg', 'model_variant': '16_bn'},
    'vgg19': {'type': 'vgg', 'model_variant': 19},
    'vgg19_bn': {'type': 'vgg', 'model_variant': '19_bn'},
    'shufflenet_v2_x0_5': {'type': 'shufflenet_v2', 'model_variant': 'x0_5'},
    'shufflenet_v2_x1_0': {'type': 'shufflenet_v2', 'model_variant': 'x1_0'},
    'shufflenet_v2_x1_5': {'type': 'shufflenet_v2', 'model_variant': 'x1_5'},
    'shufflenet_v2_x2_0': {'type': 'shufflenet_v2', 'model_variant': 'x2_0'},
    'squeezenet1_0': {'type': 'squeezenet', 'model_variant': '1_0'},
    'squeezenet1_1': {'type': 'squeezenet', 'model_variant': '1_1'},
    'swin_t': {'type': 'swin_transformer', 'model_variant': 't'},
    'swin_s': {'type': 'swin_transformer', 'model_variant': 's'},
    'swin_b': {'type': 'swin_transformer', 'model_variant': 'b'},
    'swin_v2_t': {'type': 'swin_transformer', 'model_variant': 'v2_t'},
    'swin_v2_s': {'type': 'swin_transformer', 'model_variant': 'v2_s'},
    'swin_v2_b': {'type': 'swin_transformer', 'model_variant': 'v2_b'},
    'vit_b_16': {'type': 'vision_transformer', 'model_variant': 'b_16'},
    'vit_b_32': {'type': 'vision_transformer', 'model_variant': 'b_32'},
    'vit_l_16': {'type': 'vision_transformer', 'model_variant': 'l_16'},
    'vit_l_32': {'type': 'vision_transformer', 'model_variant': 'l_32'},
    'vit_h_14': {'type': 'vision_transformer', 'model_variant': 'h_14'},
    'convnext_tiny': {'type': 'convnext', 'model_variant': 'tiny'},
    'convnext_small': {'type': 'convnext', 'model_variant': 'small'},
    'convnext_base': {'type': 'convnext', 'model_variant': 'base'},
    'convnext_large': {'type': 'convnext', 'model_variant': 'large'},
    'maxvit_t': {'type': 'maxvit', 'model_variant': 't'},
    'alexnet': {'type': 'alexnet'},
    'googlenet': {'type': 'googlenet'},
    'inception_v3': {'type': 'inception_v3'},
    'mobilenet_v2': {'type': 'mobilenet_v2'},
    'mobilenet_v3_large': {'type': 'mobilenet_v3_large'},
    'mobilenet_v3_small': {'type': 'mobilenet_v3_small'},
}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger("ImageLearner")

def split_data_0_2(
    df: pd.DataFrame,
    split_column: str,
    validation_size: float = 0.25,
    random_state: int = 42,
    label_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert a DataFrame with split values {0, 2} into {0, 1, 2}, where
    0 = train, 1 = validation, 2 = test.
    """
    df_copy = df.copy()
    train_mask = df_copy[split_column] == 0
    train_df = df_copy[train_mask]

    if train_df.empty:
        logger.info("No training data (split=0) found; skipping validation split.")
        return df_copy

    # Determine if we can stratify by label
    stratify = None
    if label_column and label_column in train_df.columns:
        unique_labels = train_df[label_column].nunique()
        if unique_labels > 1:
            min_count = train_df[label_column].value_counts().min()
            if 0 < validation_size < 1 and min_count * validation_size >= 1 and min_count * (1 - validation_size) >= 1:
                stratify = train_df[label_column]
                logger.info(f"Stratifying train/validation split by '{label_column}'.")
            else:
                logger.warning(
                    "Validation size or label distribution too small for stratification; proceeding without stratify."
                )

    # Handle edge validation_size cases
    if validation_size >= 1:
        df_copy.loc[train_mask, split_column] = 1
        logger.info("All train data moved to validation (validation_size >= 1).")
        return df_copy

    if validation_size <= 0:
        logger.info("No validation split created (validation_size <= 0).")
        return df_copy

    # Perform train/validation split
    try:
        train_idx, val_idx = train_test_split(
            train_df.index,
            test_size=validation_size,
            random_state=random_state,
            stratify=stratify,
        )
        df_copy.loc[train_idx, split_column] = 0
        df_copy.loc[val_idx, split_column] = 1
        logger.info(f"Split train data into train and validation (size={validation_size}).")
    except ValueError as e:
        logger.warning(f"Stratified split failed ({e}); retrying without stratify.")
        train_idx, val_idx = train_test_split(
            train_df.index,
            test_size=validation_size,
            random_state=random_state,
            stratify=None,
        )
        df_copy.loc[train_idx, split_column] = 0
        df_copy.loc[val_idx, split_column] = 1
        logger.info("Performed non-stratified train/validation split.")

    return df_copy


class Backend(Protocol):
    """Interface for a machine learning backend."""
    def prepare_config(
        self,
        config_params: Dict[str, Any],
        split_config: Dict[str, Any]
    ) -> str:
        ...
    def run_experiment(
        self,
        dataset_path: Path,
        config_path: Path,
        output_dir: Path,
    ) -> None:
        ...
    
    def generate_plots(
        self,
        output_dir: Path
    ) -> None:
        ...
    
    def generate_html_report(
        self,
        title: str
    ) -> None:
        ...


class LudwigDirectBackend:
    """
    Backend for running Ludwig experiments directly via the internal experiment_cli function.
    """

    def prepare_config(
        self,
        config_params: Dict[str, Any],
        split_config: Dict[str, Any],
    ) -> str:
        """
        Build and serialize the Ludwig YAML configuration.
        """
        logger.info("LudwigDirectBackend: Preparing YAML configuration.")

        model_name = config_params.get("model_name", "resnet18")
        use_pretrained = config_params.get("use_pretrained", False)
        fine_tune = config_params.get("fine_tune", False)
        epochs = config_params.get("epochs", 10)
        batch_size = config_params.get("batch_size")
        num_processes = config_params.get("preprocessing_num_processes", 1)

        # Encoder setup
        raw_encoder = MODEL_ENCODER_TEMPLATES.get(model_name, model_name)
        if isinstance(raw_encoder, dict):
            encoder_config = {
                **raw_encoder,
                "use_pretrained": use_pretrained,
                "trainable": fine_tune,
            }
        else:
            encoder_config = {"type": raw_encoder}

        # Trainer & optimizer
        # optimizer = {"type": "adam", "learning_rate": 5e-5} if fine_tune else {"type": "adam"}
        batch_size_cfg = batch_size or "auto"

        conf: Dict[str, Any] = {
            "model_type": "ecd",
            "input_features": [
                {
                    "name": IMAGE_PATH_COLUMN_NAME,
                    "type": "image",
                    "encoder": encoder_config,
                }
            ],
            "output_features": [
                {"name": LABEL_COLUMN_NAME, "type": "category"}
            ],
            "combiner": {"type": "concat"},
            "trainer": {
                "epochs": epochs,
                "early_stop": 5,
                "batch_size": batch_size_cfg,
            },
            "preprocessing": {
                "split": split_config,
                "num_processes": num_processes,
                "in_memory": False,
            },
        }

        logger.debug("LudwigDirectBackend: Config dict built.")
        try:
            yaml_str = yaml.dump(conf, sort_keys=False, indent=2)
            logger.info("LudwigDirectBackend: YAML config generated.")
            return yaml_str
        except Exception:
            logger.error("LudwigDirectBackend: Failed to serialize YAML.", exc_info=True)
            raise

    def run_experiment(
        self,
        dataset_path: Path,
        config_path: Path,
        output_dir: Path,
    ) -> None:
        """
        Invoke Ludwig's internal experiment_cli function to run the experiment.
        """
        logger.info("LudwigDirectBackend: Starting experiment execution.")

        # if not dataset_path.is_file():
        #     raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        # if not config_path.is_file():
        #     raise FileNotFoundError(f"Config file not found: {config_path}")

        # Dynamic import to avoid import errors if Ludwig is not needed
        try:
            from ludwig.experiment import experiment_cli
        except ImportError as e:
            logger.error(
                "LudwigDirectBackend: Could not import experiment_cli.", 
                exc_info=True
            )
            raise RuntimeError("Ludwig import failed.") from e

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            experiment_cli(
                dataset=str(dataset_path),
                config=str(config_path),
                output_directory=str(output_dir),
            )
            logger.info(f"LudwigDirectBackend: Experiment completed. Results in {output_dir}")
        except TypeError as e:
            logger.error(
                "LudwigDirectBackend: Argument mismatch in experiment_cli call.",
                exc_info=True
            )
            raise RuntimeError("Ludwig argument error.") from e
        except Exception:
            logger.error(
                "LudwigDirectBackend: Experiment execution error.",
                exc_info=True
            )
            raise

    def generate_plots(self, output_dir: Path) -> None:
        """
        Generate _all_ registered Ludwig visualizations for the latest experiment run.
        """
        logger.info("Generating all Ludwig visualizations…")

        # 1) find the most recent experiment directory
        exp_dirs = sorted(
            output_dir.glob("experiment*"),
            key=lambda p: p.stat().st_mtime
        )
        if not exp_dirs:
            logger.warning(f"No experiment run dirs found in {output_dir}")
            return
        exp_dir = exp_dirs[-1]

        # 2) ensure viz output subfolder exists
        viz_dir = exp_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # 3) helper to check file existence
        def _check(p: Path) -> Optional[str]:
            return str(p) if p.exists() else None

        # 4) gather standard Ludwig output files
        training_stats = _check(exp_dir / "training_statistics.json")
        test_stats     = _check(exp_dir / TEST_STATISTICS_FILE_NAME)
        probs_path     = _check(exp_dir / PREDICTIONS_PARQUET_FILE_NAME)
        gt_metadata    = _check(exp_dir / "model" / TRAIN_SET_METADATA_FILE_NAME)

        # 5) try to read original dataset & split file from description.json
        dataset_path = None
        split_file   = None
        desc = exp_dir / DESCRIPTION_FILE_NAME
        if desc.exists():
            with open(desc, "r") as f:
                cfg = json.load(f)
            dataset_path = _check(Path(cfg.get("dataset", "")))
            split_file   = _check(Path(get_split_path(cfg.get("dataset", ""))))

        # 6) infer output feature name
        output_feature = ""
        if desc.exists():
            try:
                output_feature = cfg["config"]["output_features"][0]["name"]
            except Exception:
                pass
        if not output_feature and test_stats:
            with open(test_stats, "r") as f:
                stats = json.load(f)
            output_feature = next(iter(stats.keys()), "")

        # 7) loop through every registered viz
        viz_registry = get_visualizations_registry()
        for viz_name, viz_func in viz_registry.items():
            try:
                viz_func(
                    training_statistics=[training_stats] if training_stats else [],
                    test_statistics=[test_stats]         if test_stats     else [],
                    probabilities=[probs_path]           if probs_path     else [],
                    output_feature_name=output_feature,
                    ground_truth_split=2,
                    top_n_classes=[0],
                    top_k=3,
                    ground_truth_metadata=gt_metadata,
                    ground_truth=dataset_path,
                    split_file=split_file,
                    output_directory=str(viz_dir),
                    normalize=False,
                    file_format="png",
                )
                logger.info(f"✔ Generated {viz_name}")
            except Exception as e:
                logger.warning(f"✘ Skipped {viz_name}: {e}")

        logger.info(f"All visualizations written to {viz_dir}")

    def generate_html_report(self, title: str) -> Path:
        """
        Assemble an HTML report of all plots under the last experiment's
        visualizations/ folder, then write it to the current working directory
        as '{title_lower}_report.html'.
        Returns the Path to the generated report.
        """
        cwd = Path.cwd()
        report_name = title.lower().replace(" ", "_") + "_report.html"
        report_path = cwd / report_name

        # 1) locate latest experiment run directory
        exp_dirs = sorted(
            report_path.parent.glob("experiment*"),
            key=lambda p: p.stat().st_mtime
        )
        if not exp_dirs:
            raise RuntimeError(f"No 'experiment*' dirs found in {report_path.parent}")
        exp_dir = exp_dirs[-1]

        viz_dir = exp_dir / "visualizations"

        # 2) start building HTML
        html = get_html_template()
        html += f"<h1>{title}</h1>"

        if viz_dir.exists() and viz_dir.is_dir():
            pngs = sorted(viz_dir.glob("*.png"))
            if pngs:
                html += "<div>"
                for img in pngs:
                    b64 = encode_image_to_base64(str(img))
                    html += (
                        f'<div class="plot" style="margin-bottom:20px;text-align:center;">'
                        f"<h3>{img.stem.replace('_',' ').title()}</h3>"
                        f'<img src="data:image/png;base64,{b64}" '
                        'style="max-width:90%;max-height:600px;border:1px solid #ddd;" />'
                        "</div>"
                    )
                html += "</div>"
            else:
                html += "<p><em>No .png files found in visualizations/</em></p>"
        else:
            html += "<p><em>Visualization directory not found.</em></p>"

        html += get_html_closing()

        # 3) write report into cwd
        try:
            with open(report_path, "w") as f:
                f.write(html)
            logger.info(f"HTML report generated at: {report_path}")
        except Exception as e:
            logger.error(f"Failed to write HTML report: {e}")
            raise

        return report_path


class WorkflowOrchestrator:
    """
    Manages the image-classification workflow:
      1. Creates temp dirs
      2. Extracts images
      3. Prepares data (CSV + splits)
      4. Renders a backend config
      5. Runs the experiment
      6. Cleans up
    """

    def __init__(self, args: argparse.Namespace, backend: Backend):
        self.args = args
        self.backend = backend
        self.temp_dir: Optional[Path] = None
        self.image_extract_dir: Optional[Path] = None
        logger.info(f"Orchestrator initialized with backend: {type(backend).__name__}")

    def _create_temp_dirs(self) -> None:
        """Create temporary output and image extraction directories."""
        try:
            self.temp_dir = Path(tempfile.mkdtemp(
                dir=self.args.output_dir,
                prefix=TEMP_DIR_PREFIX
            ))
            self.image_extract_dir = self.temp_dir / "images"
            self.image_extract_dir.mkdir()
            logger.info(f"Created temp directory: {self.temp_dir}")
        except Exception:
            logger.error("Failed to create temporary directories", exc_info=True)
            raise

    def _extract_images(self) -> None:
        """Extract images from ZIP into the temp image directory."""
        if self.image_extract_dir is None:
            raise RuntimeError("Temp image directory not initialized.")
        logger.info(f"Extracting images from {self.args.image_zip} → {self.image_extract_dir}")
        try:
            with zipfile.ZipFile(self.args.image_zip, "r") as z:
                z.extractall(self.image_extract_dir)
            logger.info("Image extraction complete.")
        except Exception:
            logger.error("Error extracting zip file", exc_info=True)
            raise

    def _prepare_data(self) -> Tuple[Path, Dict[str, Any]]:
        """
        Load CSV, update image paths, handle splits, and write prepared CSV.
        Returns:
            final_csv_path: Path to the prepared CSV
            split_config: Dict for backend split settings
        """
        if not self.temp_dir or not self.image_extract_dir:
            raise RuntimeError("Temp dirs not initialized before data prep.")

        # 1) Load
        try:
            df = pd.read_csv(self.args.csv_file)
            logger.info(f"Loaded CSV: {self.args.csv_file}")
        except Exception:
            logger.error("Error loading CSV file", exc_info=True)
            raise

        # 2) Validate columns
        required = {IMAGE_PATH_COLUMN_NAME, LABEL_COLUMN_NAME}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(missing)}")

        # 3) Update image paths
        try:
            df[IMAGE_PATH_COLUMN_NAME] = df[IMAGE_PATH_COLUMN_NAME].apply(
                lambda p: str((self.image_extract_dir / p).resolve())
            )
        except Exception:
            logger.error("Error updating image paths", exc_info=True)
            raise

        # 4) Handle splits
        if SPLIT_COLUMN_NAME in df.columns:
            split_config = self._process_fixed_split(df)
        else:
            logger.info("No split column; using random split")
            split_config = {
                "type": "random",
                "probabilities": self.args.split_probabilities
            }

        # 5) Write out prepared CSV
        final_csv = TEMP_CSV_FILENAME
        try:
            df.to_csv(final_csv, index=False)
            logger.info(f"Saved prepared data to {final_csv}")
        except Exception:
            logger.error("Error saving prepared CSV", exc_info=True)
            raise

        return final_csv, split_config

    def _process_fixed_split(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Process a fixed split column (0=train,1=val,2=test)."""
        logger.info(f"Fixed split column '{SPLIT_COLUMN_NAME}' detected.")
        try:
            col = df[SPLIT_COLUMN_NAME]
            df[SPLIT_COLUMN_NAME] = pd.to_numeric(col, errors="coerce").astype(pd.Int64Dtype())
            if df[SPLIT_COLUMN_NAME].isna().any():
                logger.warning("Split column contains non-numeric/missing values.")

            unique = set(df[SPLIT_COLUMN_NAME].dropna().unique())
            logger.info(f"Unique split values: {unique}")

            if unique == {0, 2}:
                df = split_data_0_2(
                    df, SPLIT_COLUMN_NAME,
                    validation_size=self.args.validation_size,
                    label_column=LABEL_COLUMN_NAME
                )
                logger.info("Applied custom 0/2 split.")
            elif unique.issubset({0, 1, 2}):
                logger.info("Using fixed split as-is.")
            else:
                raise ValueError(f"Unexpected split values: {unique}")

            return {"type": "fixed", "column": SPLIT_COLUMN_NAME}

        except Exception:
            logger.error("Error processing fixed split", exc_info=True)
            raise

    def _cleanup_temp_dirs(self) -> None:
        """Remove any temporary directories."""
        if self.temp_dir and self.temp_dir.exists():
            logger.info(f"Cleaning up temp directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir = None
        self.image_extract_dir = None

    def run(self) -> None:
        """Execute the full workflow end-to-end."""
        logger.info("Starting workflow...")
        self.args.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._create_temp_dirs()
            self._extract_images()
            csv_path, split_cfg = self._prepare_data()

            use_pretrained = False
            if self.args.use_pretrained:
                use_pretrained = True
            else:
                if self.args.fine_tune:
                    use_pretrained = True

            backend_args = {
                "model_name": self.args.model_name,
                "fine_tune": self.args.fine_tune,
                "use_pretrained": use_pretrained,
                "epochs": self.args.epochs,
                "batch_size": self.args.batch_size,
                "preprocessing_num_processes": self.args.preprocessing_num_processes,
            }
            yaml_str = self.backend.prepare_config(backend_args, split_cfg)

            config_file = self.temp_dir / TEMP_CONFIG_FILENAME
            config_file.write_text(yaml_str)
            logger.info(f"Wrote backend config: {config_file}")

            self.backend.run_experiment(csv_path, config_file, self.args.output_dir)
            logger.info("Workflow completed successfully.")
            self.backend.generate_plots(self.args.output_dir)
            report_file = self.backend.generate_plots("Image Classification Results") 
            logger.info(f"HTML report generated at: {report_file}")
        except Exception:
            logger.error("Workflow execution failed", exc_info=True)
            raise

        finally:
            self._cleanup_temp_dirs()


def main():

    parser = argparse.ArgumentParser(
        description="Image Classification Learner with Pluggable Backends"
    )
    parser.add_argument(
        "--csv-file", required=True, type=Path,
        help="Path to the input CSV"
    )
    parser.add_argument(
        "--image-zip", required=True, type=Path,
        help="Path to the images ZIP"
    )
    parser.add_argument(
        "--model-name", required=True,
        choices=MODEL_ENCODER_TEMPLATES.keys(),
        help="Which model template to use"
    )
    parser.add_argument(
        "use_pretrained", action="store_true",
        help="Use pretrained weights for the model"
    )
    parser.add_argument(
        "--fine-tune", action="store_true",
        help="Enable fine-tuning"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int,
        help="Batch size (None = auto)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("learner_output"),
        help="Where to write outputs"
    )
    parser.add_argument(
        "--validation-size", type=float, default=0.25,
        help="Fraction for validation (0.0–1.0)"
    )
    parser.add_argument(
        "--preprocessing-num-processes", type=int,
        default=max(1, os.cpu_count() // 2),
        help="CPU processes for data prep"
    )

    args = parser.parse_args()

    # -- Validation --
    if not 0.0 <= args.validation_size <= 1.0:
        parser.error("validation-size must be between 0.0 and 1.0")
    if not args.csv_file.is_file():
        parser.error(f"CSV not found: {args.csv_file}")
    if not args.image_zip.is_file():
        parser.error(f"ZIP not found: {args.image_zip}")

    # --- Instantiate Backend and Orchestrator ---
    # Use the new LudwigDirectBackend
    backend_instance = LudwigDirectBackend()
    orchestrator = WorkflowOrchestrator(args, backend_instance)

    # --- Run Workflow ---
    exit_code = 0
    try:
        orchestrator.run()
        logger.info("Main script finished successfully.")
    except Exception as e:
        logger.error("Main script failed.")
        exit_code = 1
    finally:
        sys.exit(exit_code)


if __name__ == '__main__':
    try: 
        import yaml
    except ImportError: 
        logger.error("PyYAML not found. pip install pyyaml")
        sys.exit(1)
    try:
        import ludwig
        logger.debug(f"Found Ludwig version: {ludwig.globals.LUDWIG_VERSION}")
    except ImportError:
        logger.error("Ludwig library not found. Please ensure Ludwig is installed ('pip install ludwig[image]')")
        sys.exit(1)

    main()