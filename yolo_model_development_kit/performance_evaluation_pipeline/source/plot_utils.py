import logging
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from yolo_model_development_kit.performance_evaluation_pipeline.metrics import (
    CategoryManager,
)

logger = logging.getLogger(__name__)


def _extract_plot_df(
    results_df: pd.DataFrame,
    split: str,
    target_class_name: str,
) -> pd.DataFrame:
    plot_df = results_df[
        (results_df["Split"] == split)
        & (results_df["Object Class"] == target_class_name)
    ].set_index("Conf")

    return plot_df


def _plot_pr_f_curve(
    plot_df: pd.DataFrame,
    title: str,
    size_sml: bool = False,
    logx: bool = False,
    show_plot: bool = False,
) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_df[plot_df["Size"] == "all"].plot(
        ax=ax,
        ylim=[0, 1],
        logx=logx,
        grid=True,
        color=["tab:green", "tab:blue", "tab:orange"],
        lw=3,
    )

    if size_sml:
        # This only makes sense for recall plots, since no other data is
        # available for different size categories.

        linestyle = {
            "small": ":",
            "medium": "-.",
            "large": "--",
        }

        for label, df in plot_df[plot_df["Size"] != "all"].groupby(by="Size"):
            legend_label = f"Recall {label}"
            df["Recall"].plot(
                ax=ax,
                xlabel="Confidence threshold",
                ylim=[0, 1],
                logx=logx,
                grid=True,
                label=legend_label,
                color="tab:blue",
                style=linestyle[label],
                lw=1.5,
            )

    plt.title(title)
    plt.legend(loc="lower left")

    if not show_plot:
        plt.close()

    return fig


def _generate_plot_title_and_filename(
    result_type: str,
    plot_type: str,
    dataset: str,
    split: str,
    model_name: str,
    target_class: str,
    filename: Optional[str] = None,
) -> Tuple[str, str]:
    title = f"{result_type.upper()}\nDataset: {dataset}_{split}, Model: {model_name}, Object: {target_class}"
    if not filename:
        filename = f"{'-'.join(result_type.split())}_{split}_{target_class.replace('_', '-')}_{plot_type}.png"
    return title, filename


def save_pr_curve(
    results_df: pd.DataFrame,
    split: str,
    target_class: int,
    category_manager: CategoryManager,
    model_name: str,
    result_type: str,
    dataset: str = "",
    output_dir: str = "",
    filename: Optional[str] = None,
    size_sml: bool = False,
    logx: bool = False,
    show_plot: bool = False,
) -> None:
    """
    Plot and save the precision and recall curve for a particular split and target_class.
    """

    target_class_name = category_manager.get_name(target_class)
    if target_class_name == "Unknown":
        raise ValueError(f"Class ID {target_class} not found in loaded categories.")

    plot_df = _extract_plot_df(
        results_df=results_df, split=split, target_class_name=target_class_name
    )[["Size", "Precision", "Recall"]]

    (title, filename) = _generate_plot_title_and_filename(
        result_type=result_type,
        plot_type="pr-curve",
        dataset=dataset,
        split=split,
        model_name=model_name,
        target_class=target_class_name,
        filename=filename,
    )

    fig = _plot_pr_f_curve(
        plot_df=plot_df,
        title=title,
        size_sml=size_sml,
        logx=logx,
        show_plot=show_plot,
    )
    fig.savefig(os.path.join(output_dir, filename), dpi=150)


def save_fscore_curve(
    results_df: pd.DataFrame,
    split: str,
    category_manager: CategoryManager,
    target_class: int,
    model_name: str,
    result_type: str,
    dataset: str = "",
    output_dir: str = "",
    filename: Optional[str] = None,
    logx: bool = False,
    show_plot: bool = False,
) -> None:
    """
    Plot and save the F-score curve for a particular split and target_class.
    """

    target_class_name = category_manager.get_name(target_class)
    if target_class_name == "Unknown":
        raise ValueError(f"Class ID {target_class} not found in loaded categories.")

    plot_df = _extract_plot_df(
        results_df=results_df, split=split, target_class_name=target_class_name
    )[["Size", "F1", "F0.5", "F2"]]

    (title, filename) = _generate_plot_title_and_filename(
        result_type=result_type,
        plot_type="f-score",
        dataset=dataset,
        split=split,
        model_name=model_name,
        target_class=target_class_name,
        filename=filename,
    )

    fig = _plot_pr_f_curve(
        plot_df=plot_df,
        title=title,
        size_sml=False,
        logx=logx,
        show_plot=show_plot,
    )
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
