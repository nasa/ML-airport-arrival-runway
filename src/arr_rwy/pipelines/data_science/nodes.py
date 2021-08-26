"""Nodes for performing data science
"""

import logging
import mlflow
import mlflow.sklearn

import pandas as pd

from copy import deepcopy
from typing import Any, Dict, List, Tuple
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline as sklearn_Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer as sklearn_ColumnTransformer

from data_services.mlflow_utils import init_mlflow_run
from data_services.FilterPipeline import FilterPipeline
from data_services.RunwayModelWrapper import RunwayModelWrapper
from data_services.transformed_target_classifier import TransformedTargetClassifier
from data_services.filter_pipeline_utils import add_rules_to_filter_pipeline
from data_services.drop_inputs_transformer import DropInputsTransfomer
from data_services.data_inspector import append_data_inspector
from data_services.OrderFeatures import OrderFeatures

log = logging.getLogger(__name__)

def assemble_feature_prep_pipeline(
        inputs: Dict[str, Any],
        sklearn_pipeline_impute_steps: List[Tuple],
        sklearn_pipeline_encode_steps: List[Tuple],
        sklearn_pipeline_transform_steps: List[Tuple],
        pipeline_inspect_data_verbosity: int = 0,
        data_inspector_verbosity: int = 0,
        ) -> List[Tuple]:
    """
    Trains a model

    Note on pipeline_inspect_data_verbosity approach
    0: no data inspectors
    1: only right before going into model
    2: same as 1, plus inspect raw input right at start
    3: after all per-input impute, after all per-input encode,
    and after all per-input transform steps
    4: between all steps
    """

    sklearn_Pipeline_steps = []

    # Always include an OrderFeatures, just to be safe
    order_features = OrderFeatures()
    sklearn_Pipeline_steps.extend(
        [
            ("order_features", order_features),
         ],
        )

    # Initial data inspector of raw input data
    if pipeline_inspect_data_verbosity > 1:
        sklearn_Pipeline_steps = append_data_inspector(
            sklearn_Pipeline_steps,
            data_inspector_verbosity,
            data_inspector_prefix='raw input data',
            data_inspector_name='data_inspector_raw_input',
        )

    # Impute
    sklearn_Pipeline_steps.extend(sklearn_pipeline_impute_steps)
    if pipeline_inspect_data_verbosity > 2:
        sklearn_Pipeline_steps = append_data_inspector(
            sklearn_Pipeline_steps,
            data_inspector_verbosity,
            data_inspector_prefix='data after imputation',
            data_inspector_name='data_inspector_post_impute',
        )

    # Encode
    sklearn_Pipeline_steps.extend(sklearn_pipeline_encode_steps)
    if pipeline_inspect_data_verbosity > 2:
        sklearn_Pipeline_steps = append_data_inspector(
            sklearn_Pipeline_steps,
            data_inspector_verbosity,
            data_inspector_prefix='data after encoding',
            data_inspector_name='data_inspector_post_encode',
        )

    # Transform (per-input)
    sklearn_Pipeline_steps.extend(sklearn_pipeline_transform_steps)
    if pipeline_inspect_data_verbosity > 2:
        sklearn_Pipeline_steps = append_data_inspector(
            sklearn_Pipeline_steps,
            data_inspector_verbosity,
            data_inspector_prefix='data after transforms',
            data_inspector_name='data_inspector_post_transform',
        )

    # Step to drop raw inputs not for model
    # (these inputs were used to calculate features,
    # but are not themselves features for the model)
    drop_input_cols = []
    for model_input in inputs:
        try:
            if inputs[model_input]['drop_before_model']:
                drop_input_cols.append(model_input)
        except KeyError:
            continue
    drop_inputs = DropInputsTransfomer(drop_input_cols)
    sklearn_Pipeline_steps.append(
        ('drop_inputs', drop_inputs)
    )

    # Make data inspector transformer to check out data going into model
    if pipeline_inspect_data_verbosity > 0:
        sklearn_Pipeline_steps = append_data_inspector(
            sklearn_Pipeline_steps,
            data_inspector_verbosity,
            data_inspector_prefix='calculated features provided to model',
            data_inspector_name='data_inspector_pre_model',
        )

    return sklearn_Pipeline_steps

def train_models(
        dat: pd.DataFrame,
        model_params: Dict[str, Dict[str, Any]],
        sklearn_Pipeline_steps: List[Tuple],
        inputs: Dict[str, Any],
        target: Dict[str, Any],
        mlflow_params: Dict[str, Any],
        experiment_id: int,
        global_pararms: Dict[str, Any],
        categories: Dict[str, List[Any]],
        ) -> Dict[str, Dict[str, Any]]:
    """
    Parameters
    ----------
    dat : pd.DataFrame
        Full test/train dataset
    model_params : Dict[str, Dict[str, Any]]
        Standard specification for all models
    sklearn_Pipeline_steps : List[Tuple]
        Steps to add to sklearn pipeline
    inputs : Dict[str, Any]
        Standard input specification
    target : Dict[str, Any]
        Standard target specification
    mlflow_params : Dict[str, Any]
        Standard MLflow connection / usage specification
    experiment_id : int
        MLflow experiment ID to use for these models
    global_pararms : Dict[str, Any]
        Full set of airport-specific global parameters
    categories : Dict[str, List[Any]]
        List of allowable values for categorical variables

    Returns
    -------
    model_pipelines : Dict[str, Dict[str, Any]]
        Dictionary of model data and objects

    """

    model_pipelines = dict()

    for model_name in model_params:
        log.info("Preparing to train / log {} model".format(
            model_name,
            ))
        model, training_stats = _train_model(
            dat,
            model_name,
            model_params[model_name],
            sklearn_Pipeline_steps,
            target,
            inputs,
            categories,
            )

        run_id = _log_model_to_mlflow(
            model,
            mlflow_params,
            experiment_id,
            model_name,
            model_params[model_name],
            inputs,
            global_pararms,
            training_stats,
            )

        model_pipelines[model_name] = {
            "model":model,
            "run_id":run_id,
            }

    return model_pipelines

def _train_model(
        dat: pd.DataFrame,
        model_name: str,
        model_params: Dict[str, Any],
        sklearn_Pipeline_steps: List[Tuple],
        target: Dict[str, Any],
        inputs: Dict[str, Any],
        categories: Dict[str, List[Any]],
        ) -> Tuple[FilterPipeline, Dict[str, Any]]:
    """
    Parameters
    ----------
    dat : pd.DataFrame
        Full test/train dataset
    model_name : str
        Name of this model
    model_params : Dict[str, Any]
        Standard specification for this model
    sklearn_Pipeline_steps : List[Tuple]
        Steps to add to sklearn pipeline
    target : Dict[str, Any]
        Standard target specification
    inputs : Dict[str, Any]
        Standard input specification
    categories : Dict[str, List[Any]]
        List of allowable values for categorical variables

    Returns
    -------
    Tuple[FilterPipeline, Dict[str, Any]]
        Trained model pipeline and dictionary of model training stats
    """

    steps_to_append = deepcopy(sklearn_Pipeline_steps)

    if (model_name == "DummyClassifier"):
        model = DummyClassifier(
            **model_params["model_params"],
            )
    elif (model_name == "LogisticRegression"):
        model = LogisticRegression(
            **model_params["model_params"],
            )
    elif (model_name == "XGBClassifier"):
        dtype_transformer = sklearn_ColumnTransformer(
            transformers=[(
                "dtype_transformer",
                "passthrough",
                make_column_selector(
                    dtype_exclude="number",
                    ),
                )],
            remainder="passthrough",
            )

        steps_to_append.append(
            ("dtype_transformer", dtype_transformer),
            )

        model = TransformedTargetClassifier(
            regressor=XGBClassifier(
                **model_params["model_params"],
                ),
            transformer=LabelEncoder(),
            check_inverse=False,
            )
    else:
        raise(Exception("Unknown model type provided"))

    steps_to_append.append(
        ("model", model),
        )

    pipeline = sklearn_Pipeline(
        steps=steps_to_append,
        )

    X_train = (
        dat.loc[
            (dat["train_sample"] == True),
            inputs.keys(),
            ]
        .copy()
        )

    y_train = (
        dat.loc[
            (dat["train_sample"] == True),
            target["name"],
            ]
        .copy()
        )

    for col in X_train.columns:
        num_nulls = X_train[col].isnull().sum()
        if (num_nulls > 0):
            log.info("column {} in train dataset has {} nulls".format(
                col,
                num_nulls,
                ))

    log.info("Unique target values are {}".format(
        ",".join([str(x) for x in y_train.unique()])
        ))

    # TODO: this does not use the reduced category list -- should it?
    default_response = y_train.mode()[0]

    log.info("Identified runway {} as default response".format(
        default_response,
        ))

    # TODO: would it be more sensible to return np.nan for default?
    fp = add_rules_to_filter_pipeline(
        RunwayModelWrapper(
            operation="arr",
            core_pipeline=pipeline,
            default_response=default_response,
            print_debug=True,
            default_score_behavior="model",
            ),
        inputs,
        target,
        categories,
        )

    log.info("Starting model training")
    tic = pd.Timestamp.now()
    fp = fp.fit(
        X_train,
        y_train,
        )
    toc = pd.Timestamp.now()
    log.info("Model training for {} with {} samples took {:.1f} seconds".format(
        model_name,
        (dat["train_sample"] == True).sum(),
        (toc - tic).total_seconds(),
        ))

    training_stats = {
        "num_training_samples":(dat["train_sample"] == True).sum(),
        "training_time":(toc - tic).total_seconds(),
        }

    return fp, training_stats

def _log_model_to_mlflow(
        model: FilterPipeline,
        mlflow_params: Dict[str, Any],
        experiment_id,
        model_name: str,
        model_params: Dict[str, Any],
        inputs: Dict[str, Any],
        global_pararms: Dict[str, Any],
        training_stats: Dict[str, Any],
        ) -> str:
    """
    Parameters
    ----------
    model : FilterPipeline
        Trained model pipeline
    mlflow_params : Dict[str, Any]
        Standard MLflow connection / usage specification
    experiment_id : int
        MLflow experiment ID to use for these models
    model_name : str
        Name of this model
    model_params : Dict[str, Any]
        Standard specification for this model
    inputs : Dict[str, Any]
        Standard input specification
    global_pararms : Dict[str, Any]
        Full set of airport-specific global parameters
    training_stats : Dict[str, Any]
        dictionary of model training stats

    Returns
    -------
    run_id : str
        Run_id to which this model was logged
    """

    run_id = init_mlflow_run({"mlflow":mlflow_params}, experiment_id)
    with mlflow.start_run(run_id=run_id):
        log.info("Logging trained {} model and params to MLFlow under run_id {}".format(
            model_name,
            run_id,
            ))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            )

        mlflow.log_param("model", model_name)

        mlflow.log_param("features", list(inputs.keys()))

        core_features = [n for n in inputs if inputs[n]["core"] == True]
        mlflow.log_param("core_features", core_features)

        mlflow.log_param("start_time", global_pararms["start_time"])
        mlflow.log_param("end_time", global_pararms["end_time"])
        mlflow.log_param("known_runways", global_pararms["known_runways"])
        mlflow.log_param("default_response", model.default_response)

        mlflow.set_tag("airport_icao", global_pararms["airport_icao"])

        for key, value in model_params.items():
            mlflow.log_param(key, value)

        for key, value in training_stats.items():
            mlflow.log_metric(key, value)

    return run_id
