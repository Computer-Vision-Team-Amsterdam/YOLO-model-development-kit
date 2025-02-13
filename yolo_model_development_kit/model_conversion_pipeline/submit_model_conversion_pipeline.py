import os

from azure.ai.ml import Output
from azure.ai.ml.dsl import pipeline

from yolo_model_development_kit import aml_interface, settings
from yolo_model_development_kit.model_conversion_pipeline.components.model_conversion import (
    convert_model,
)


@pipeline()
def model_conversion_pipeline():
    datastore_path = aml_interface.get_datastore_full_path(
        settings["model_conversion"]["datastore_path"]
    )
    model_weights_rel_path = settings["model_conversion"]["model_weights_rel_path"]
    model_weights_dir = os.path.join(datastore_path, model_weights_rel_path)

    print("Model conversion pipeline starting...")
    run_model_conversion_step = convert_model()
    run_model_conversion_step.outputs.model_weights_dir = Output(
        type="uri_folder", mode="rw_mount", path=model_weights_dir
    )

    return {}


def main() -> None:
    default_compute = settings["aml_experiment_details"]["compute_name"]
    aml_interface.submit_pipeline_experiment(
        model_conversion_pipeline, "model_conversion_pipeline", default_compute
    )


if __name__ == "__main__":
    main()
