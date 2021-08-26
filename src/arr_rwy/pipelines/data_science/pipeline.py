"""Pipeline for arrival runway prediction data science
"""

from kedro.pipeline import Pipeline, node

from data_services.mlflow_utils import init_mlflow
from data_services.network import copy_artifacts_to_ntx
from data_services.filter_pipeline_utils import get_feature_categories
from data_services.feature_prep_utils import (
    assemble_impute_steps,
    assemble_encode_steps,
    assemble_transform_steps,
    )
from data_services.runway_scoring import score_models

from .nodes import (
    assemble_feature_prep_pipeline,
    train_models,
    )

def create_pipelines(**kwargs):
    train_pipeline = Pipeline(
        nodes=[

            node(
                func=init_mlflow,
                inputs=[
                    "parameters",
                    ],
                outputs="experiment_id",
                name="init_mlflow_ds"
                ),

            node(
                func=get_feature_categories,
                inputs=[
                    "de_data_set@CSV",
                    "params:inputs",
                    "params:target",
                    ],
                outputs="categories",
                name="get_feature_categories",
                ),

            node(
                func=assemble_impute_steps,
                inputs=[
                    'params:inputs',
                    'params:pipeline_inspect_data_verbosity',
                    'params:data_inspector_verbosity',
                    ],
                outputs='sklearn_pipeline_impute_steps',
                name='assemble_impute_steps',
                ),

            node(
                func=assemble_encode_steps,
                inputs={
                    "dat":'de_data_set@CSV',
                    "inputs":'params:inputs',
                    "pipeline_inspect_data_verbosity":'params:pipeline_inspect_data_verbosity',
                    "data_inspector_verbosity":'params:data_inspector_verbosity',
                    "all_categories":"categories",
                    "known_runways":"params:globals.known_runways",
                    },
                outputs='sklearn_pipeline_encode_steps',
                name='assemble_encode_steps',
                ),

            node(
                func=assemble_transform_steps,
                inputs=[
                    'params:inputs',
                    'params:pipeline_inspect_data_verbosity',
                    'params:data_inspector_verbosity',
                    ],
                outputs='sklearn_pipeline_transform_steps',
                name='assemble_transform_steps',
                ),

            node(
                func=assemble_feature_prep_pipeline,
                inputs={
                    "inputs":"params:inputs",
                    "sklearn_pipeline_impute_steps":
                        "sklearn_pipeline_impute_steps",
                    "sklearn_pipeline_encode_steps":
                        "sklearn_pipeline_encode_steps",
                    "sklearn_pipeline_transform_steps":
                        "sklearn_pipeline_transform_steps",
                    "pipeline_inspect_data_verbosity":
                        "params:pipeline_inspect_data_verbosity",
                    "data_inspector_verbosity":
                        "params:data_inspector_verbosity",
                    },
                outputs="feature_prep_pipeline",
                name="assemble_feature_prep_pipeline",
                ),

            node(
                func=train_models,
                inputs=[
                    "de_data_set@CSV",
                    "params:models",
                    "feature_prep_pipeline",
                    "params:inputs",
                    "params:target",
                    "params:mlflow",
                    "experiment_id",
                    "params:globals",
                    "categories",
                    ],
                outputs="model_pipelines",
                name="train_models",
                ),

            node(
                func=score_models,
                inputs=[
                    "de_data_set@CSV",
                    "model_pipelines",
                    "params:inputs",
                    "params:target",
                    "params:globals.airport_icao",
                    "categories",
                    ],
                outputs="artifacts_ready",
                name="score_models",
                ),

            # node(
            #     func=copy_artifacts_to_ntx,
            #     inputs=[
            #         "experiment_id",
            #         "model_pipelines",
            #         "params:ntx_connection",
            #         "artifacts_ready",
            #         "params:ntx_credentials.key_path",
            #         ],
            #     outputs=None,
            #     name="copy_artifacts_to_ntx",
            #     ),

            ],
        tags="ds",
        )

    return train_pipeline
