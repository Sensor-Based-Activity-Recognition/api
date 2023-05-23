import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import stft as stft_scipy


class Pipeline:
    def __init__(self):
        pass

    def run(
        self,
        data,
        model="CNN",
    ):
        match model:
            case "CNN":
                data = self.cast(data)
                data = self.resample(data, 100, "linear")
                data = self.resample(data, 50, "linear")
                windows = self.segmentate(data, 5, 0)
                stft = self.stft(windows)
                return stft
            case "HGBC":
                data = self.cast(data)
                data = self.resample(data, 100, "linear")
                data = self.resample(data, 50, "linear")
                windows = self.segmentate(data, 5, 0)
                fft = self.fft(windows)
                return fft
            case _:
                raise ValueError(f"Model {model} not supported")

    def cast(self, data):
        data["timestamp"] = pd.to_datetime(data["timestamp"])

        data = data.groupby(
            pd.to_datetime(data["timestamp"].astype("int") // 10000000 * 10000000)
        ).mean()

        data = data.drop(columns=["timestamp"])

        data = data.sort_values(by="timestamp")

        return data

    def resample(self, data, resample_frequency_hz, interpolation_method):
        data = data.resample(
            f"{int(1E6/resample_frequency_hz)}us", origin="start"
        ).interpolate(method=interpolation_method)

        # backward fill in case of missing values at start
        na_by_col = data.isna().sum()
        for col in na_by_col:
            if col > 1:
                print(
                    f"Warning: recording has more than 1 NA values in column with index {col}. Backward filling."
                )
        data = data.fillna(method="bfill")

        # convert RangeIndex to DatetimeIndex
        data.index = pd.to_datetime(data.index)

        return data

    def segmentate(self, data, window_len_s, overlap_percent):
        def segmentate_helper(df, window_len_s, overlap_percent):
            overlap_timedelta = pd.Timedelta(
                (window_len_s / 100) * overlap_percent, "s"
            )

            # add segment id column
            data.loc[:, "segment_id"] = -1
            segment_id = 0

            windows = []
            window_start = df.index[0]
            while True:
                window_end = window_start + pd.Timedelta(window_len_s, "s")

                # window cannot reach full length anymore
                if window_end > df.index[-1]:
                    return windows

                selected_rows = (df.index >= window_start) & (df.index <= window_end)

                # add segment id
                df.loc[selected_rows, "segment_id"] = segment_id
                segment_id += 1

                # segmentate dataframe and make a copy
                df_window = df.loc[selected_rows].copy()
                df_window["segment_id"] = df_window["segment_id"].astype(
                    "category"
                )  # make segment_id categorial so filters / fft / etc dont alter this column

                windows.append(df_window)
                window_start = window_end - overlap_timedelta

        # segmentate data
        return segmentate_helper(data, window_len_s, overlap_percent)

    def stft(self, data):
        def stft_helper(df):
            spectogram = []
            # for each numeric column
            for col in df.select_dtypes(include=np.number).columns:
                _, _, Zxx = stft_scipy(df[col], fs=50, noverlap=95, nperseg=100)
                spectogram.append(np.abs(Zxx))
            segment_id = df["segment_id"].iloc[0]
            return (segment_id, np.array(spectogram))

        return [stft_helper(segment) for segment in data]

    def fft(self, data):
        def fft_helper(df):
            # assert constant frequency
            assert (np.diff(df.index.values) == np.diff(df.index.values)[0]).all()

            fs = (
                pd.Timedelta(1, "s") / np.diff(df.index.values)[0]
            )  # get sampling frequency

            N = len(df)  # get length of dataframe
            xf = fftfreq(N, 1 / fs)[: N // 2]  # calculate fft sample frequencies

            # generate spectrum from each column
            data = {}
            for col in df.select_dtypes(
                include=["number"]
            ).columns:  # only select numeric columns
                yf = fft(df[col].values)

                data[col] = 2.0 / N * np.abs(yf[0 : N // 2])

            df_transformed = pd.DataFrame(
                data=data, index=xf
            )  # create dataframe a with sample frequencies on index and corresponding fft on columns

            # copy all non numerical values into dataframe
            for col in df.select_dtypes(exclude=["number"]).columns:
                series: np.array = df[col].values  # copy column

                if (
                    len(np.unique(series)) != 1
                ):  # if series has more than one distinct value
                    raise Exception(
                        "a non transformed column has more than one unique value... to prevent unexpected behavior, the transformation was stoped"
                    )

                df_transformed[col] = series[: len(xf)]  # truncate values from column

                df_transformed[col] = df_transformed[col].astype(
                    df[col].dtype
                )  # set correct datatype

            df_transformed = df_transformed.drop(columns=["segment_id"])
            df_transformed = df_transformed.to_numpy().transpose().flatten()
            return df_transformed  # return dataframe

        return np.array([fft_helper(segment) for segment in data])
