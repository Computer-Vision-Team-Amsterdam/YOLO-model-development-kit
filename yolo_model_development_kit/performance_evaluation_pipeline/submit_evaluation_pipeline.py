import os

from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from yolo_model_development_kit import aml_interface, settings
from yolo_model_development_kit.performance_evaluation_pipeline.components import (
    evaluate_model,
    perform_bias_analysis,
)


@pipeline()
def evaluation_pipeline():

    input_datastore_path = aml_interface.get_datastore_full_path(
        settings["performance_evaluation"]["inputs"]["datastore"]
    )
    ground_truth_rel_path = settings["performance_evaluation"]["inputs"][
        "ground_truth_rel_path"
    ]
    predictions_rel_path = settings["performance_evaluation"]["inputs"][
        "predictions_rel_path"
    ]
    output_datastore_path = aml_interface.get_datastore_full_path(
        settings["performance_evaluation"]["outputs"]["datastore"]
    )
    output_rel_path = settings["performance_evaluation"]["outputs"]["output_rel_path"]

    is_bias_analysis = settings["performance_evaluation"]["is_bias_analysis"]

    ground_truth_data = Input(
        type=AssetTypes.URI_FOLDER,
        path=os.path.join(input_datastore_path, ground_truth_rel_path),
    )
    predictions_data = Input(
        type=AssetTypes.URI_FOLDER,
        path=os.path.join(input_datastore_path, predictions_rel_path),
    )
    output_path = os.path.join(output_datastore_path, output_rel_path)

    if is_bias_analysis:
        bias_analysis_step = perform_bias_analysis(
            predictions_base_dir=predictions_data
        )
        bias_analysis_step.outputs.ground_truth_base_dir = Output(
            type="uri_folder",
            mode="rw_mount",
            path=os.path.join(input_datastore_path, ground_truth_rel_path),
        )
        bias_analysis_step.outputs.output_dir = Output(
            type="uri_folder",
            mode="rw_mount",
            path=output_path,
        )
    else:
        evaluate_step = evaluate_model(
            ground_truth_base_dir=ground_truth_data,
            predictions_base_dir=predictions_data,
        )
        evaluate_step.outputs.output_dir = Output(
            type="uri_folder",
            mode="rw_mount",
            path=output_path,
        )

    return {}


def main() -> None:
    default_compute = settings["aml_experiment_details"]["compute_name"]
    aml_interface.submit_pipeline_experiment(
        evaluation_pipeline, "evaluation_pipeline", default_compute
    )


if __name__ == "__main__":
    main()
