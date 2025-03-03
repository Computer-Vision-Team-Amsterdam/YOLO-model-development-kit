from yolo_model_development_kit import aml_interface, settings


def main():
    """
    This file creates an AML environment.
    """
    aml_interface.create_aml_environment(
        env_name=settings["aml_experiment_details"]["env_name"],
        build_context_path="yolo_model_development_kit/create_aml_environment",
        dockerfile_path="Dockerfile",
    )


if __name__ == "__main__":
    main()
