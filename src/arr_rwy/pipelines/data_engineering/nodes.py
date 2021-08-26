"""Nodes for performing data engineering
"""

import logging

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

def df_fill_runway_times(
        df: pd.DataFrame,
        ) -> pd.DataFrame:
    # TODO: maybe parameterize this, since the suffixes are too?
    df["arrival_runway_actual_time"] = (
        df["arrival_runway_actual_time_mfs"]
        .fillna(
            df["arrival_runway_actual_time_via_surveillance"])
        )
    return df

def start_tv_df(
        ntv_df: pd.DataFrame,
        tv_timestep: str="30s",
        ) -> pd.DataFrame:
    """
    Parameters
    ----------
    ntv_df : pd.DataFrame
        DESCRIPTION.
    tv_timestep : str, optional
        DESCRIPTION. The default is "30s".

    Returns
    -------
    tv_df : pd.DataFrame
        DESCRIPTION.

    """
    time_step = pd.to_timedelta(tv_timestep)
    
    tv_df_dict = {
        "gufi": [],
        "timestamp": [],
    }

    for _, row in ntv_df.iterrows():
        if (pd.notnull(row["departure_runway_actual_time"])
            & pd.notnull(row["arrival_runway_actual_time"])
            ):
            time_range = pd.date_range(
                start = np.maximum(
                    row["departure_runway_actual_time"],
                    row["arrival_runway_actual_time"]
                    - pd.to_timedelta(4, unit="hours")
                    ),
                end = row["arrival_runway_actual_time"] + time_step,
                freq=time_step,
            )
    
            tv_df_dict["timestamp"].extend(time_range)
            tv_df_dict["gufi"].extend([row["gufi"]] * len(time_range))

    tv_df = pd.DataFrame.from_dict(tv_df_dict)

    tv_df = tv_df.sort_values(by="timestamp")

    return tv_df
