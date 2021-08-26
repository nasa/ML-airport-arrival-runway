"""Pipeline for arrival runway prediction data engineering
"""

from kedro.pipeline import Pipeline, node

from data_services.swim_based_eta import build_swim_eta

from .nodes import (
    df_fill_runway_times,
    start_tv_df,
    )

from data_services.runway_data_engineering import (
    df_join,
    infer_wake_categories,
    sort_timestamp_merge_asof,
    add_difference_columns,
    select_tv_train_samples,
    drop_cols_not_in_inputs,
    clean_runway_names,
    drop_rows_with_missing_target,
    apply_live_filtering,
    de_save,
    )

def create_pipelines(**kwargs):
    ntv_pipeline = Pipeline(
        nodes=[

            node(
                func=df_join,
                inputs=[
                    "mfs_data_set@CSV",
                    "runways_data_set@CSV",
                    "params:df_join_kwargs.mfs_data_set.runways_data_set",
                    ],
                outputs="ntv_df_0",
                name="df_join_runways_onto_MFS",
                ),

            node(
                func=df_fill_runway_times,
                inputs=[
                    "ntv_df_0",
                    ],
                outputs="ntv_df_1",
                name="df_fill_runway_times",
                ),

            node(
                func=infer_wake_categories,
                inputs=[
                    "ntv_df_1",
                    "aircraft_types_xml@CSV",
                    ],
                outputs="ntv_df_2",
                name="infer_wake_categories",
                ),

            node(
                func=sort_timestamp_merge_asof,
                inputs={
                    "data_0":"ntv_df_2",
                    "data_1":"config_data_set@CSV",
                    "merge_asof_kwargs":"params:df_join_kwargs.ntv_df.config_data_set",
                },
                outputs="ntv_df",
                name="merge_asof_config_onto_ntv_df",
                ),

            ],
        tags="de",
        )

    tv_pipeline = Pipeline(
        nodes=[

            node(
                func=build_swim_eta,
                inputs=[
                    "tfms_etas_data_set@CSV",
                    "tbfm_stas_data_set@CSV",
                    "first_time_tracked_data_set@CSV",
                    ],
                outputs="best_swim_eta",
                name="build_swim_eta",
                ),

            node(
                func=start_tv_df,
                inputs=[
                    "ntv_df",
                    "params:tv_timestep",
                    ],
                outputs="tv_df_0",
                name="start_tv_df",
                ),

            node(
                func=sort_timestamp_merge_asof,
                inputs=[
                    "tv_df_0",
                    "best_swim_eta",
                    ],
                outputs="tv_df_1",
                name="merge_asof_swim_eta_onto_tv_df",
                ),

            node(
                func=sort_timestamp_merge_asof,
                inputs=[
                    "tv_df_1",
                    "planned_fix_data_set@CSV",
                    ],
                outputs="tv_df_2",
                name="merge_asof_planned_fix_onto_tv_df",
                ),

            node(
                func=sort_timestamp_merge_asof,
                inputs=[
                    "tv_df_2",
                    "runway_assigned_data_set@CSV",
                    ],
                outputs="tv_df",
                name="merge_asof_assigned_runway_onto_tv_df",
                ),

            ],
        tags="de",
        )
    
    final_pipeline = Pipeline(
        nodes=[

            node(
                func=df_join,
                inputs=[
                    "tv_df",
                    "ntv_df",
                    "params:df_join_kwargs.tv_df.ntv_df",
                    ],
                outputs="tv_ntv_df",
                name="df_join_ntv_onto_tv",
                ),

            node(
                func=add_difference_columns,
                inputs=[
                    "tv_ntv_df",
                    "params:difference_columns",
                    ],
                outputs="tv_ntv_df_1",
                name="compute_difference_columns",
                ),

            node(
                func=drop_cols_not_in_inputs,
                inputs=[
                    "tv_ntv_df_1",
                    "params:inputs",
                    "params:target",
                    ],
                outputs="tv_ntv_filtered",
                name="drop_cols_not_in_inputs_final_df",
                ),

            node(
                func=drop_rows_with_missing_target,
                inputs=[
                    "tv_ntv_filtered",
                    "params:target",
                    ],
                outputs="tv_ntv_filtered_cleaned",
                name="drop_rows_with_missing_target",
                ),

            node(
                func=clean_runway_names,
                inputs=[
                    "tv_ntv_filtered_cleaned",
                    "params:runway_cols_to_clean",
                    ],
                outputs="tv_ntv_clean_runways",
                name="clean_runway_names",
                ),

            node(
                func=apply_live_filtering,
                inputs=[
                    "tv_ntv_clean_runways",
                    "params:inputs",
                    ],
                outputs="tv_ntv_clean",
                name="apply_live_filtering",
                ),


            node(
                func=select_tv_train_samples,
                inputs=[
                    "tv_ntv_clean",
                    "params:test_size",
                    "params:tv_timestep_fraction_train",
                    "params:random_seed",
                    ],
                outputs="tv_ntv_df_filtered_train_sampled",
                name="select_train_samples",
                ),

            node(
                func=de_save,
                inputs=[
                    "tv_ntv_df_filtered_train_sampled",
                    "params:globals",
                    ],
                outputs=None,
                name="save_de_dataset",
                ),

            ],
        tags="de",
        )

    return {
        "de_ntv": ntv_pipeline,
        "de_tv": ntv_pipeline + tv_pipeline,
        "de_overall": ntv_pipeline + tv_pipeline + final_pipeline,
        }
