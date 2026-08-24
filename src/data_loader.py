"""
Data Ingestion and Standardized Loading Module.

Reads raw CSV files with varied schema definitions, resolves column aliases,
and standardizes timestamps and athlete identifiers.
"""

import os
from typing import Dict, Optional
import pandas as pd

from src.config import RAW_FILES, DEFAULT_OBS_END
from src.utils import find_col, LOGGER


class RawDataLoader:
    """
    Handles loading, alias resolution, and type standardization of PLAYHACK CSVs.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def _get_file_path(self, key: str) -> str:
        filename = RAW_FILES.get(key, f"{key}.csv")
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            if os.path.exists(self.data_dir):
                for actual_file in os.listdir(self.data_dir):
                    if actual_file.lower() == filename.lower():
                        return os.path.join(self.data_dir, actual_file)
        return path

    def load_daily_activity(self) -> pd.DataFrame:
        """Loads dailyActivity_merged.csv and standardizes athlete_id and date."""
        path = self._get_file_path("daily_activity")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Daily activity file not found at: {path}")

        LOGGER.info(f"Loading daily activity data from {path}...")
        df = pd.read_csv(path)

        id_col = find_col(df, "athlete_id")
        date_col = find_col(df, "activity_date")

        df = df.rename(columns={id_col: "athlete_id", date_col: "date"})
        df["athlete_id"] = df["athlete_id"].astype(int)
        return df

    def load_hourly_heartrate(self) -> pd.DataFrame:
        """Loads hourlyHeartrate_merged.csv."""
        path = self._get_file_path("hourly_heartrate")
        if not os.path.exists(path):
            LOGGER.warning(f"Hourly heart rate file not found at {path}, returning empty DataFrame.")
            return pd.DataFrame()

        LOGGER.info(f"Loading hourly heart rate data from {path}...")
        df = pd.read_csv(path)

        id_col = find_col(df, "athlete_id")
        date_col = find_col(df, "activity_date")

        df = df.rename(columns={id_col: "athlete_id", date_col: "datetime"})
        df["date"] = df["datetime"].str.slice(0, 10)
        df["athlete_id"] = df["athlete_id"].astype(int)
        return df

    def load_sleep_day(self) -> pd.DataFrame:
        """Loads sleepDay_merged.csv."""
        path = self._get_file_path("sleep_day")
        if not os.path.exists(path):
            LOGGER.warning(f"Sleep day file not found at {path}, returning empty DataFrame.")
            return pd.DataFrame()

        LOGGER.info(f"Loading sleep day data from {path}...")
        df = pd.read_csv(path)

        id_col = find_col(df, "athlete_id")
        date_col = find_col(df, "activity_date")

        df = df.rename(columns={id_col: "athlete_id", date_col: "date"})
        df["athlete_id"] = df["athlete_id"].astype(int)
        return df

    def load_training_sessions(self) -> pd.DataFrame:
        """Loads training_sessions.csv."""
        path = self._get_file_path("training_sessions")
        if not os.path.exists(path):
            LOGGER.warning(f"Training sessions file not found at {path}, returning empty DataFrame.")
            return pd.DataFrame()

        LOGGER.info(f"Loading training sessions data from {path}...")
        df = pd.read_csv(path)

        id_col = find_col(df, "athlete_id")
        date_col = find_col(df, "activity_date")

        df = df.rename(columns={id_col: "athlete_id", date_col: "date"})
        df["athlete_id"] = df["athlete_id"].astype(int)
        return df

    def load_weight_logs(self) -> pd.DataFrame:
        """Loads weightLogInfo_merged.csv."""
        path = self._get_file_path("weight_log")
        if not os.path.exists(path):
            LOGGER.warning(f"Weight log file not found at {path}, returning empty DataFrame.")
            return pd.DataFrame()

        LOGGER.info(f"Loading weight logs from {path}...")
        df = pd.read_csv(path)

        id_col = find_col(df, "athlete_id")
        date_col = find_col(df, "activity_date")

        df = df.rename(columns={id_col: "athlete_id", date_col: "date"})
        df["athlete_id"] = df["athlete_id"].astype(int)
        return df

    def load_athlete_metadata(self) -> pd.DataFrame:
        """Loads athlete_metadata.csv."""
        path = self._get_file_path("athlete_metadata")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Athlete metadata file not found at {path}")

        LOGGER.info(f"Loading athlete metadata from {path}...")
        df = pd.read_csv(path)

        id_col = find_col(df, "athlete_id")
        df = df.rename(columns={id_col: "athlete_id"})
        df["athlete_id"] = df["athlete_id"].astype(int)
        return df

    def load_train_labels(self, default_obs_end: Optional[str] = DEFAULT_OBS_END) -> pd.DataFrame:
        """
        Loads train_labels.csv. Resolves targets and observation window end anchor.
        """
        path = self._get_file_path("train_labels")
        if not os.path.exists(path):
            LOGGER.warning(f"Train labels file not found at {path}.")
            return pd.DataFrame()

        LOGGER.info(f"Loading training labels from {path}...")
        df = pd.read_csv(path)

        id_col = find_col(df, "athlete_id")
        df = df.rename(columns={id_col: "athlete_id"})
        df["athlete_id"] = df["athlete_id"].astype(int)

        # Standardize target names if present
        target_col = find_col(df, "injured_in_risk_window", required=False)
        if target_col and target_col != "injured_in_risk_window":
            df = df.rename(columns={target_col: "injured_in_risk_window"})

        onset_col = find_col(df, "onset_day_offset", required=False)
        if onset_col and onset_col != "onset_day_offset":
            df = df.rename(columns={onset_col: "onset_day_offset"})

        rec_col = find_col(df, "recovery_duration", required=False)
        if rec_col and rec_col != "recovery_duration":
            df = df.rename(columns={rec_col: "recovery_duration"})

        obs_col = find_col(df, "obs_window_end", required=False)
        if obs_col:
            df["obs_window_end"] = df[obs_col].astype(str).str.slice(0, 10)
        elif default_obs_end is not None:
            df["obs_window_end"] = str(default_obs_end)[:10]
        else:
            raise KeyError(
                "Could not find the observation-window end date.\n"
                "Expected one of:\n"
                "  - obs_window_end\n"
                "  - observation_window_end\n"
                "  - obs_end\n"
                "  - window_end\n"
                "Please provide an observation window anchor."
            )

        return df


def load_all_raw_data(
    data_dir: str,
    default_obs_end: Optional[str] = DEFAULT_OBS_END,
) -> Dict[str, pd.DataFrame]:
    """
    Loads all core raw tables for feature extraction.
    """
    loader = RawDataLoader(data_dir)
    return {
        "daily_activity": loader.load_daily_activity(),
        "hourly_heartrate": loader.load_hourly_heartrate(),
        "sleep_day": loader.load_sleep_day(),
        "training_sessions": loader.load_training_sessions(),
        "weight_logs": loader.load_weight_logs(),
        "athlete_metadata": loader.load_athlete_metadata(),
        "train_labels": loader.load_train_labels(default_obs_end=default_obs_end),
    }
