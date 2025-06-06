import logging
import os
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from yolo_model_development_kit.performance_evaluation_pipeline.metrics import (
    CategoryManager,
    PerImageEvaluator,
    PerPixelEvaluator,
    compute_fb_score,
)
from yolo_model_development_kit.performance_evaluation_pipeline.source.plot_utils import (
    save_fscore_curve,
    save_pr_curve,
)
from yolo_model_development_kit.performance_evaluation_pipeline.source.run_custom_coco_eval import (
    execute_custom_coco_eval,
)
from yolo_model_development_kit.performance_evaluation_pipeline.source.yolo_to_coco import (
    convert_yolo_dataset_to_coco_json,
    convert_yolo_predictions_to_coco_json,
)

logger = logging.getLogger("performance_evaluation")


class YoloEvaluator:
    """
    This class is used to run evaluation of a trained YOLO model based on ground
    truth annotations and model predictions.

    YoloEvaluator supports three evaluation methods:

    * Total Blurred Area evaluation for sensitive classes. This tells us the
      percentage of bounding boxes that are covered by predictions.

    * Per Image evaluation. This tells us the precision and recall based on
      whole images, i.e. if a single image contains at least one annotation of a
      certain class, does it also contain at least one prediction.

    * Custom COCO evaluation. This is a COCO-style evaluation of overall and per
      class precision and recall, for different bounding box sizes and
      confidence thresholds.

    Results are returned as Dictionaries that can optionally be converted to
    DataFrames.

    Parameters
    ----------

    ground_truth_base_folder: str
        Location of ground truth dataset (root folder, is expected to contain
        `images/` and `labels/` subfolders).
    predictions_base_folder: str
        Location of predictions (root folder, is expected to contain `labels/`
        subfolder).
    category_manager: CategoryManager
        CategoryManager object containing object classes.
    output_folder: Optional[str] = None
        Location where output will be stored. If None, the
        predictions_base_folder will be used.
    predictions_image_shape: Tuple[int, int] = (3840, 2160)
        Shape of prediction images as (w, h).
    dataset_name: str = ""
        Name of dataset, used in results plots.
    model_name: Optional[str] = None
        Name of the model used in the results. If no name is provided, the name
        of the predictions folder is used.
    gt_annotations_rel_path: str = "labels"
        Name of folder containing ground truth labels.
    pred_annotations_rel_path: str = "labels"
        Name of the folder containing prediction labels.
    splits: Optional[List[str]] = ["train", "val", "test"]
        Which splits to evaluate. Set to `None` if the data contains no splits.
    target_classes: List[int] = []
        Which object classes should be evaluated (default is []).
    sensitive_classes: List[int] = []
        Which object classes should be treated as sensitive for the Total
        Blurred Area computation (default is ["person", "license_plate"]).
    target_classes_conf: Optional[float] = None
        Confidence threshold used for target classes. If not specified, all
        predictions will be evaluated.
    sensitive_classes_conf: Optional[float] = None
        Confidence threshold used for sensitive classes. If not specified, all
        predictions will be evaluated.
    single_size_only: bool = False
        Set to true to disable differentiation in bounding box sizes. Default is
        to evaluate for the sizes S, M, and L.
    plot_sml_size: bool = False
        Whether to plot PR curves for all bounding boxes combined (False), or
        differentiate by size (True).
    plot_conf_range: Optional[Iterable[float]] = None
        Range of confidence values over which to plot PR/F curves. If not set,
        range will be taken as 0.05 intervals between 0 and 1.
    plot_logx: Optional[bool] = False
        Whether to use log scale for plot x-axis
    """

    def __init__(
        self,
        ground_truth_base_folder: str,
        predictions_base_folder: str,
        category_manager: CategoryManager,
        output_folder: Optional[str] = None,
        predictions_image_shape: Tuple[int, int] = (3840, 2160),
        dataset_name: str = "",
        model_name: Optional[str] = None,
        gt_annotations_rel_path: str = "labels",
        pred_annotations_rel_path: str = "labels",
        splits: Optional[List[str]] = ["train", "val", "test"],
        target_classes: List[int] = [],
        sensitive_classes: List[int] = [],
        target_classes_conf: Optional[float] = None,
        sensitive_classes_conf: Optional[float] = None,
        overall_stats_tba: bool = False,
        single_size_only: bool = False,
        plot_sml_size: bool = False,
        plot_conf_range: Optional[Iterable[float]] = None,
        plot_logx: Optional[bool] = False,
    ):
        self.ground_truth_base_folder = ground_truth_base_folder
        self.predictions_base_folder = predictions_base_folder
        self.category_manager = category_manager
        self.output_folder = output_folder
        self.predictions_image_shape = predictions_image_shape
        self.dataset_name = dataset_name
        self.model_name = (
            model_name
            if model_name
            else os.path.basename(os.path.dirname(predictions_base_folder))
        )
        self.gt_annotations_rel_path = gt_annotations_rel_path
        self.pred_annotations_rel_path = pred_annotations_rel_path
        self.splits = splits if splits else [""]

        self.target_classes = target_classes
        self.sensitive_classes = sensitive_classes
        self.all_classes = self.target_classes + self.sensitive_classes

        self.target_classes_conf = target_classes_conf
        self.sensitive_classes_conf = sensitive_classes_conf
        self.overall_stats_tba = overall_stats_tba
        self.single_size_only = single_size_only

        if plot_conf_range is not None:
            self.plot_conf_range = plot_conf_range
        else:
            self.plot_conf_range = np.arange(0.05, 1.0, 0.05)
        self.plot_sml_size = (not self.single_size_only) and plot_sml_size
        self.plot_logx = plot_logx

        self._log_stats()

    def _log_stats(self) -> None:
        """Log number of annotation files in ground truth and prediction folders
        as sanity check."""
        for split in self.splits:
            split_name = split if split != "" else "all"
            gt_folder, pred_folder = self._get_folders_for_split(split)
            gt_count = len(
                [name for name in os.listdir(gt_folder) if name.endswith(".txt")]
            )
            pred_count = len(
                [name for name in os.listdir(pred_folder) if name.endswith(".txt")]
            )
            logger.info(
                f"Split: {split_name}, ground truth labels: {gt_count}, prediction labels: {pred_count}"
            )

    def _get_folders_for_split(self, split: str) -> Tuple[str, str]:
        """Generate the full path to ground truth and prediction annotation
        folders for a specific split."""
        ground_truth_folder = os.path.join(
            self.ground_truth_base_folder, self.gt_annotations_rel_path, split
        )
        prediction_folder = os.path.join(
            self.predictions_base_folder, self.pred_annotations_rel_path, split
        )
        return ground_truth_folder, prediction_folder

    def evaluate_tba(
        self,
        upper_half: bool = False,
        confidence_threshold: Optional[float] = None,
        single_size_only: Optional[bool] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Run Total Blurred Area evaluation for the sensitive classes. This tells
        us the percentage of bounding boxes that are covered by predictions.

        The results are summarized in a dictionary as follows:

            {
                [model_name]_[split]: {
                    [object_class]_[size]: {
                        "true_positives": float,
                        "false_positives": float,
                        "true_negatives": float,
                        "false_negatives:": float,
                        "precision": float,
                        "recall": float,
                        "f1_score": float,
                    }
                }
            }

        Parameters
        ----------
        upper_half: bool = False
            Whether to only consider the upper half of bounding boxes (relevant
            for people, to make sure the face is blurred).
        confidence_threshold: Optional[float] = None
            Optional: confidence threshold at which to compute statistics. If
            omitted, the initial confidence threshold at construction will be
            used.
        single_size_only: Optional[bool] = None
            Optional: set to true to disable differentiation in bounding box
            sizes. If omitted, the initial confidence threshold at construction
            will be used.

        Returns
        -------
        Results as Dict[str, Dict[str, Dict[str, float]]] as described above.
        """
        if not confidence_threshold:
            confidence_threshold = self.sensitive_classes_conf
        if not single_size_only:
            single_size_only = self.single_size_only

        tba_results = dict()
        for split in self.splits:
            logger.info(
                f"Running TBA evaluation for {self.model_name} / {split if split != '' else 'all'} "
                f"@ {confidence_threshold} confidence"
            )
            ground_truth_folder, prediction_folder = self._get_folders_for_split(split)
            evaluator = PerPixelEvaluator(
                ground_truth_path=ground_truth_folder,
                predictions_path=prediction_folder,
                category_manager=self.category_manager,
                image_shape=self.predictions_image_shape,
                confidence_threshold=confidence_threshold,
                upper_half=upper_half,
            )
            key = f"{self.model_name}_{split if split != '' else 'all'}"
            tba_results[key] = evaluator.collect_results_per_class_and_size(
                classes=self.sensitive_classes,
                single_size_only=single_size_only,
                include_overall_stats=self.overall_stats_tba,
            )
        return tba_results

    def evaluate_tba_bias_analysis(
        self,
        grouping: Dict[str, Dict[str, Any]],
        upper_half: bool = False,
        confidence_threshold: Optional[float] = None,
        single_size_only: Optional[bool] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Run Total Blurred Area evaluation for the sensitive classes and for a specific grouping.
        This tells us the percentage of bounding boxes that are covered by predictions.

        The results are summarized in a dictionary as follows:

            {
                [model_name]_[split]: {
                    [object_class]_[size]: {
                        "true_positives": float,
                        "false_positives": float,
                        "true_negatives": float,
                        "false_negatives:": float,
                        "precision": float,
                        "recall": float,
                        "f1_score": float,
                    }
                }
            }

        Parameters
        ----------
        upper_half: bool = False
            Whether to only consider the upper half of bounding boxes (relevant
            for people, to make sure the face is blurred).
        confidence_threshold: Optional[float] = None
            Optional: confidence threshold at which to compute statistics. If
            omitted, the initial confidence threshold at construction will be
            used.
        single_size_only: Optional[bool] = None
            Optional: set to true to disable differentiation in bounding box
            sizes. If omitted, the initial confidence threshold at construction
            will be used.

        Returns
        -------
        Results as Dict[str, Dict[str, Dict[str, float]]] as described above.
        """
        if not confidence_threshold:
            confidence_threshold = self.sensitive_classes_conf
        if not single_size_only:
            single_size_only = self.single_size_only

        tba_results = dict()
        for split in self.splits:
            logger.info(
                f"Running TBA bias analysis for {self.model_name} / {split if split != '' else 'all'} "
                f"@ {confidence_threshold} confidence"
            )
            ground_truth_folder, prediction_folder = self._get_folders_for_split(split)
            evaluator = PerPixelEvaluator(
                ground_truth_path=ground_truth_folder,
                predictions_path=prediction_folder,
                category_manager=self.category_manager,
                image_shape=self.predictions_image_shape,
                confidence_threshold=confidence_threshold,
                upper_half=upper_half,
            )
            key_prefix = f"{self.model_name}_{split if split != '' else 'all'}"

            group_mapping = {
                int(grouping["maps_to"]["class"][0]): [
                    cat["category_id"] for cat in grouping["categories"].values()
                ]
            }

            logger.info(f"Group mapping: {group_mapping}")

            for target_class, group_categories in group_mapping.items():
                for category in group_categories:
                    logger.info(
                        f"Running TBA bias analysis for class {target_class} vs category {category}"
                    )
                    key = f"{key_prefix}_class_{target_class}_vs_{category}"
                    tba_results[key] = evaluator.collect_results_per_class_and_size(
                        classes=[target_class],
                        single_size_only=single_size_only,
                        use_group_mapping=True,
                        group_mapping={target_class: [category]},
                    )

        return tba_results

    def evaluate_per_image(
        self,
        confidence_threshold: Optional[float] = None,
        single_size_only: Optional[bool] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Run Per Image evaluation for the sensitive classes. This tells us the
        precision and recall based on whole images, i.e. if a single image
        contains at least one annotation of a certain class, does it also
        contain at least one prediction.

        The results are summarized in a dictionary as follows:

            {
                [model_name]_[split]: {
                    [object_class]_[size]: {
                        "precision": float,
                        "recall": float,
                        "fpr": float,
                        "fnr": float,
                        "tnr": float,
                    }
                }
            }

        Parameters
        ----------
        confidence_threshold: Optional[float] = None
            Optional: confidence threshold at which to compute statistics. If
            omitted, the initial confidence threshold at construction will be
            used.
        single_size_only: Optional[bool] = None
            Optional: set to true to disable differentiation in bounding box
            sizes. If omitted, the initial confidence threshold at construction
            will be used.

        Returns
        -------
        Results as Dict[str, Dict[str, Dict[str, float]]] as described above.
        """
        if not confidence_threshold:
            confidence_threshold = self.target_classes_conf
        if not single_size_only:
            single_size_only = self.single_size_only

        per_image_results = dict()
        for split in self.splits:
            logger.info(
                f"Running per-image evaluation for {self.model_name} / {split if split != '' else 'all'} "
                f"@ {confidence_threshold} confidence"
            )
            ground_truth_folder, prediction_folder = self._get_folders_for_split(split)
            evaluator = PerImageEvaluator(
                ground_truth_path=ground_truth_folder,
                predictions_path=prediction_folder,
                category_manager=self.category_manager,
                image_shape=self.predictions_image_shape,
                confidence_threshold=confidence_threshold,
            )
            key = f"{self.model_name}_{split if split != '' else 'all'}"
            per_image_results[key] = evaluator.collect_results_per_class_and_size(
                classes=self.target_classes, single_size_only=single_size_only
            )
        return per_image_results

    def evaluate_coco(
        self, confidence_threshold: Optional[float] = None
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Run custom COCO evaluation. This is a COCO-style evaluation of overall
        and per class precision and recall, for different bounding box sizes and
        confidence thresholds.

        TODO: come up with a way to incorporate the confidence threshold.

        The results are summarized in a dictionary as follows:

            {
                [model_name]_[split]: {
                    [object_class]: {
                        "AP@50-95_all": float,
                        "AP@75_all": float,
                        "AP@50_all": float,
                        "AP@50_small": float,
                        "AP@50_medium": float,
                        "AP@50_large": float,
                        "AR@50-95_all": float,
                        "AR@75_all": float,
                        "AR@50_all": float,
                        "AR@50_small": float,
                        "AR@50_medium": float,
                        "AR@50_large": float,
                    }
                }
            }

        Parameters
        ----------
        confidence_threshold: Optional[float] = None
            Optional: confidence threshold at which to compute statistics. If
            omitted, the initial confidence threshold at construction will be
            used.

        Returns
        -------
        Results as Dict[str, Dict[str, Dict[str, float]]] as described above.
        """
        if not confidence_threshold:
            confidence_threshold = self.target_classes_conf

        custom_coco_result: Dict[str, Dict[str, Dict[str, float]]] = dict()
        coco_eval_classes = {"all": self.all_classes}
        for class_id in self.all_classes:
            class_name = self.category_manager.get_name(class_id)
            coco_eval_classes[class_name] = [class_id]

        # The custom COCO evaluation needs annotations in COCO JSON format, so we need to convert.
        ## Set output folders for COCO JSON files.
        if not self.output_folder:
            gt_output_dir = self.ground_truth_base_folder
        else:
            gt_output_dir = self.output_folder
        if not self.output_folder:
            pred_output_dir = self.predictions_base_folder
        else:
            pred_output_dir = self.output_folder
        ## Run conversion.
        gt_json_files = convert_yolo_dataset_to_coco_json(
            dataset_dir=self.ground_truth_base_folder,
            category_manager=self.category_manager,
            splits=self.splits,
            fixed_image_shape=self.predictions_image_shape,
            output_dir=gt_output_dir,
            gt_labels_rel_path=self.gt_annotations_rel_path,
        )
        pred_json_files = convert_yolo_predictions_to_coco_json(
            predictions_dir=self.predictions_base_folder,
            image_shape=self.predictions_image_shape,
            labels_rel_path=self.pred_annotations_rel_path,
            splits=self.splits,
            output_dir=pred_output_dir,
            conf=(confidence_threshold if confidence_threshold else 0.0),
        )

        # Run evaluation
        for i, split in enumerate(self.splits):
            key = f"{self.model_name}_{split if split != '' else 'all'}"
            custom_coco_result[key] = dict()

            for target_cls_name, target_cls in coco_eval_classes.items():
                logger.info(
                    f"Running custom COCO evaluation for {self.model_name} / {split if split != '' else 'all'} / {target_cls_name}"
                )
                eval = execute_custom_coco_eval(
                    coco_ground_truth_json=gt_json_files[i],
                    coco_predictions_json=pred_json_files[i],
                    category_manager=self.category_manager,
                    predicted_img_shape=self.predictions_image_shape,
                    classes=target_cls,
                    print_summary=False,
                )
                subkey = target_cls_name
                custom_coco_result[key][subkey] = eval

        # Remove temporary JSON files
        for gt_file, pred_file in zip(gt_json_files, pred_json_files):
            os.remove(gt_file)
            os.remove(pred_file)

        return custom_coco_result

    def save_tba_results_to_csv(
        self,
        results: Dict[str, Dict[str, Dict[str, float]]],
        use_groupings: bool = False,
        group_id: Optional[int] = None,
    ):
        """Save TBA results dict as CSV file."""
        filename = ""
        if use_groupings:
            if group_id is None:
                raise ValueError("group_id must be provided when use_groupings=True.")
            if len(self.target_classes) != 1:
                raise ValueError(
                    f"Expected exactly one target class, got: {self.target_classes}."
                )
            target_classes_str = f"{self.target_classes[0]}"
            filename = os.path.join(
                self.output_folder,
                f"{self.model_name}-{target_classes_str}-{group_id}-tba-eval.csv",
            )
            _df_to_csv(bias_analysis_tba_result_to_df(results), filename)
        else:
            filename = os.path.join(
                self.output_folder, f"{self.model_name}-tba-eval.csv"
            )
            _df_to_csv(tba_result_to_df(results), filename)

    def save_per_image_results_to_csv(
        self, results: Dict[str, Dict[str, Dict[str, float]]]
    ):
        """Save per-image results dict as CSV file."""
        filename = os.path.join(
            self.output_folder, f"{self.model_name}-per-image-eval.csv"
        )
        _df_to_csv(per_image_result_to_df(results), filename)

    def save_coco_results_to_csv(self, results: Dict[str, Dict[str, Dict[str, float]]]):
        """Save COCO results dict as CSV file."""
        filename = os.path.join(
            self.output_folder, f"{self.model_name}-custom-coco-eval.csv"
        )
        _df_to_csv(custom_coco_result_to_df(results), filename)

    def _compute_pr_f_curve_data(self, eval_func: Callable) -> pd.DataFrame:
        """
        Compute data needed to plot precision and recall curves, and the f-score
        curves. This method calls either `self.evaluate_tba` or
        `self.evaluate_per_image` with argument `single_size_only=True` for a
        range of confidence thresholds and returns the results in a DataFrame.

        Parameters
        ----------
        eval_func: Callable
            Either `self.evaluate_tba` or `self.evaluate_per_image`.

        Returns
        -------
        A pandas DataFrame with the precision, recall, F1, F0.5, and F2 scores
        for each confidence level.
        """
        dfs = []

        for conf in self.plot_conf_range:
            logger.debug(f"Computing PR/F curve data for conf={conf}")
            tba_results = eval_func(
                confidence_threshold=conf, single_size_only=(not self.plot_sml_size)
            )
            df = tba_result_to_df(tba_results)
            df.insert(4, "Conf", conf)
            df["F1"] = compute_fb_score(df["Precision"], df["Recall"], 1.0)
            df["F0.5"] = compute_fb_score(df["Precision"], df["Recall"], 0.5)
            df["F2"] = compute_fb_score(df["Precision"], df["Recall"], 2.0)
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    def _plot_pr_f_curves(
        self,
        pr_df: pd.DataFrame,
        result_type: str,
        eval_classes: List[int],
        output_dir: str = "",
        show_plot: bool = False,
    ):
        """Plot the precision and recall curves, and F-score curves."""
        extended_eval_classes: List[Union[int, str]] = [
            cls_id for cls_id in eval_classes
        ]
        if self.overall_stats_tba:
            extended_eval_classes.insert(0, "all")
        for split, eval_class in product(self.splits, extended_eval_classes):
            save_pr_curve(
                results_df=pr_df,
                split=(split if split != "" else "all"),
                target_class=eval_class,
                category_manager=self.category_manager,
                model_name=self.model_name,
                result_type=result_type,
                dataset=self.dataset_name,
                output_dir=output_dir,
                size_sml=self.plot_sml_size,
                logx=self.plot_logx,
                show_plot=show_plot,
            )
            save_fscore_curve(
                results_df=pr_df,
                split=(split if split != "" else "all"),
                target_class=eval_class,
                category_manager=self.category_manager,
                model_name=self.model_name,
                result_type=result_type,
                dataset=self.dataset_name,
                output_dir=output_dir,
                logx=self.plot_logx,
                show_plot=show_plot,
            )

    def plot_tba_pr_f_curves(self, show_plot: bool = False):
        """
        Plot and save precision and recall curves and f-score curves for the
        total blurred area statistic. This will call evaluate_tba() for each
        split and target_class for a range of confidence thresholds, which can
        take some time to compute.

        Parameters
        ----------
        show_plot: bool = False
            Whether or not to show the plot (True) or only save the image
            (False).
        """
        logger.info(f"Plotting TBA precision/recall curves for {self.model_name}")
        pr_curve_df = self._compute_pr_f_curve_data(self.evaluate_tba)
        self._plot_pr_f_curves(
            pr_df=pr_curve_df,
            result_type="total blurred area",
            eval_classes=self.sensitive_classes,
            output_dir=self.output_folder,
            show_plot=show_plot,
        )
        # Also save results to csv
        filename = os.path.join(self.output_folder, "tba-pr-f-curve-data.csv")
        _df_to_csv(pr_curve_df, filename)

    def plot_per_image_pr_f_curves(self, show_plot: bool = False):
        """
        Plot and save precision and recall curves and f-score curves for the
        per-image statistic. This will call evaluate_per-image() for a each
        split and target_class for a range of confidence thresholds, which can
        take some time to compute.

        Parameters
        ----------
        show_plot: bool = False
            Whether or not to show the plot (True) or only save the image
            (False).
        """
        logger.info(f"Plotting per-image precision/recall curves for {self.model_name}")
        pr_curve_df = self._compute_pr_f_curve_data(self.evaluate_per_image)
        self._plot_pr_f_curves(
            pr_df=pr_curve_df,
            result_type="per image",
            eval_classes=self.target_classes,
            output_dir=self.output_folder,
            show_plot=show_plot,
        )
        # Also save results to csv
        filename = os.path.join(self.output_folder, "per-image-pr-f-curve-data.csv")
        _df_to_csv(pr_curve_df, filename)


def _bias_analysis_result_to_df(
    results: Dict[str, Dict[str, Dict[str, float]]],
) -> pd.DataFrame:
    """
    Convert bias analysis results dictionary to Pandas DataFrame.
    """

    def _stat_to_header(stat: str) -> str:
        """For nicer column headings we transform 'true_positives' -> 'True Positives' etc."""
        if stat in ("fpr", "fnr", "tpr", "tnr"):
            return stat.upper()
        else:
            parts = [p.capitalize() for p in stat.split(sep="_")]
            return " ".join(parts)

    models = list(results.keys())
    statistics = list(results[models[0]].values())[0].keys()

    header = [
        "Model",
        "Split",
        "Object Class",
        "Size",
        "Ground Truth Class",
    ]
    header.extend([_stat_to_header(stat) for stat in statistics])

    df = pd.DataFrame(columns=header)

    for model in models:
        # Splitting based on the known format of the key: [model_name]_[split]_class_[target_class]_vs_[group_id]
        try:
            model_name, split, _, target_class, _, group_id = model.rsplit(
                "_", maxsplit=5
            )
        except ValueError:
            raise ValueError(f"Unexpected model key format: {model}")

        categories = results[model].keys()

        for cat in categories:
            if cat not in results[model]:
                raise KeyError(
                    f"Expected key '{cat}' not found in results for model '{model}'."
                )

            try:
                _, ground_truth_class, size = cat.split("_", maxsplit=2)
            except ValueError:
                raise ValueError(f"Unexpected category key format: {cat}")

            # Create a row of data for each category
            data: List[Any] = [
                model_name,  # Model
                split if split != "all" else "",  # Split (empty if "all")
                target_class,  # Object Class
                size,  # Size (all, small, medium, large)
                ground_truth_class,  # Ground Truth Class
            ]
            data.extend([val for val in results[model][cat].values()])
            df.loc[len(df)] = data

    return df


def _default_result_to_df(
    results: Dict[str, Dict[str, Dict[str, float]]],
) -> pd.DataFrame:
    """
    Convert TBA or PerImage results dictionary to Pandas DataFrame.
    """

    def _stat_to_header(stat: str) -> str:
        """For nicer column headings we transform 'true_positives' -> 'True Positives' etc."""
        if stat in ("fpr", "fnr", "tpr", "tnr"):
            return stat.upper()
        else:
            parts = [p.capitalize() for p in stat.split(sep="_")]
            return " ".join(parts)

    models = list(results.keys())
    categories = list(results[models[0]].keys())
    statistics = list(results[models[0]][categories[0]].keys())
    header = [
        "Model",
        "Split",
        "Object Class",
        "Size",
    ]
    header.extend([_stat_to_header(stat) for stat in statistics])

    df = pd.DataFrame(columns=header)

    for model in models:
        (model_name, split) = model.rsplit(sep="_", maxsplit=1)
        for cat in categories:
            (cat_name, size) = cat.rsplit(sep="_", maxsplit=1)
            data: List[Any] = [model_name, split, cat_name, size]
            data.extend([val for val in results[model][cat].values()])
            df.loc[len(df)] = data

    return df


def bias_analysis_tba_result_to_df(
    results: Dict[str, Dict[str, Dict[str, float]]],
) -> pd.DataFrame:
    """
    Convert TBA results dictionary to Pandas DataFrame.
    """
    return _bias_analysis_result_to_df(results=results)


def tba_result_to_df(results: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    """
    Convert TBA results dictionary to Pandas DataFrame.
    """
    return _default_result_to_df(results=results)


def per_image_result_to_df(
    results: Dict[str, Dict[str, Dict[str, float]]],
) -> pd.DataFrame:
    """
    Convert Per Image results dictionary to Pandas DataFrame.
    """
    return _default_result_to_df(results=results)


def custom_coco_result_to_df(
    results: Dict[str, Dict[str, Dict[str, float]]],
) -> pd.DataFrame:
    """
    Convert custom COCO results dictionary to Pandas DataFrame.
    """
    models = list(results.keys())
    categories = list(results[models[0]].keys())
    header = ["Model", "Split", "Object Class"]
    header.extend(list(results[models[0]][categories[0]].keys()))

    df = pd.DataFrame(columns=header)

    for model in models:
        (model_name, split) = model.rsplit(sep="_", maxsplit=1)
        for cat in categories:
            data: List[Any] = [model_name, split, cat]
            data.extend([val for val in results[model][cat].values()])
            df.loc[len(df)] = data

    return df


def _df_to_csv(df: pd.DataFrame, output_file: str):
    """Convenience method, currently not very useful but allows to change
    formatting of all CSVs in one place."""
    df.to_csv(output_file)
