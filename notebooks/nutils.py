import sys

sys.path.append("../")
from numba import jit
import numpy as np
import pandas as pd


@jit(nogil=True, nopython=True)
def _time_diff(timestamps, a, diff_time, larger=True):
    assert a.shape[0] == timestamps.shape[0]

    result_len = a.shape[0]

    if abs(diff_time) < 1e4:
        result = np.full(result_len, np.nan)
        if diff_time > 0:
            for i in range(diff_time, result_len):
                if timestamps[i] - timestamps[i - diff_time] > 60 * 60 * 10**9:
                    result[i] = np.nan
                else:
                    result[i] = a[i] - a[i - diff_time]
        elif diff_time < 0:
            for i in range(result_len + diff_time):
                if timestamps[i - diff_time] - timestamps[i] > 60 * 60 * 10**9:
                    result[i] = np.nan
                else:
                    result[i] = a[i] - a[i - diff_time]
        else:
            result = np.zeros_like(a, dtype=np.float64)

    else:
        result = np.zeros_like(a, dtype=np.float64)
        if larger:
            if diff_time > 0:
                j = -1
                for i in range(result_len):
                    while timestamps[i] - timestamps[j + 1] >= diff_time:
                        j += 1
                    if j >= 0 and timestamps[i] - timestamps[j] < 60 * 60 * 10**9:
                        result[i] = a[i] - a[j]
                    else:
                        result[i] = np.nan

            elif diff_time < 0:
                j = 0
                for i in range(result_len):
                    while j < result_len and timestamps[j] - timestamps[i] < -diff_time:
                        j += 1
                    if j != result_len:
                        if timestamps[j] - timestamps[i] >= 60 * 60 * 10**9:
                            result[i] = np.nan
                        else:
                            result[i] = a[i] - a[j]
                    else:
                        result[i] = np.nan

            else:
                pass
        else:
            if diff_time > 0:
                j = -1
                for i in range(result_len):
                    while timestamps[i] - timestamps[j + 1] > diff_time:
                        j += 1
                    if j >= 0 and timestamps[i] - timestamps[j] < 60 * 60 * 10**9:
                        result[i] = a[i] - a[j + 1]
                    else:
                        result[i] = np.nan

            elif diff_time < 0:
                j = 0
                for i in range(result_len):
                    while (
                        j < result_len and timestamps[j] - timestamps[i] <= -diff_time
                    ):
                        j += 1
                    if j != result_len:
                        if timestamps[j] - timestamps[i] >= 60 * 60 * 10**9:
                            result[i] = np.nan
                        else:
                            result[i] = a[i] - a[j - 1]
                    else:
                        result[i] = np.nan

            else:
                pass

    return result


def time_diff(timestamps, a, diff_time):
    if hasattr(timestamps, "values"):
        timestamps = timestamps.values
    if hasattr(a, "values"):
        a = a.values
    diff_time = int(diff_time)

    return _time_diff(timestamps, a, diff_time)


@jit(nogil=True, error_model="numpy")
def _log_return(timestamps, a, diff_time, larger=True):
    assert a.shape[0] == timestamps.shape[0]

    result_len = a.shape[0]

    if abs(diff_time) < 1e4:
        result = np.full(result_len, np.nan)
        if diff_time > 0:
            for i in range(diff_time, result_len):
                if timestamps[i] - timestamps[i - diff_time] > 60 * 60 * 10**9:
                    result[i] = np.nan
                else:
                    result[i] = np.log(a[i] / a[i - diff_time])
        elif diff_time < 0:
            for i in range(result_len + diff_time):
                if timestamps[i - diff_time] - timestamps[i] > 60 * 60 * 10**9:
                    result[i] = np.nan
                else:
                    result[i] = np.log(a[i] / a[i - diff_time])
        else:
            result = np.zeros_like(a, dtype=np.float64)

    else:
        result = np.zeros_like(a, dtype=np.float64)
        if larger:
            if diff_time > 0:
                j = -1
                for i in range(result_len):
                    while timestamps[i] - timestamps[j + 1] >= diff_time:
                        j += 1
                    if j >= 0 and timestamps[i] - timestamps[j] < 60 * 60 * 10**9:
                        result[i] = np.log(a[i] / a[j])
                    else:
                        result[i] = np.nan

            elif diff_time < 0:
                j = 0
                for i in range(result_len):
                    while j < result_len and timestamps[j] - timestamps[i] < -diff_time:
                        j += 1
                    if j != result_len:
                        if timestamps[j] - timestamps[i] >= 60 * 60 * 10**9:
                            result[i] = np.nan
                        else:
                            result[i] = np.log(a[i] / a[j])
                    else:
                        result[i] = np.nan

            else:
                pass
        else:
            if diff_time > 0:
                j = -1
                for i in range(result_len):
                    while timestamps[i] - timestamps[j + 1] > diff_time:
                        j += 1
                    if j >= 0 and timestamps[i] - timestamps[j] < 60 * 60 * 10**9:
                        result[i] = np.log(a[i] / a[j + 1])
                    else:
                        result[i] = np.nan

            elif diff_time < 0:
                j = 0
                for i in range(result_len):
                    while (
                        j < result_len and timestamps[j] - timestamps[i] <= -diff_time
                    ):
                        j += 1
                    if j != result_len:
                        if timestamps[j] - timestamps[i] >= 60 * 60 * 10**9:
                            result[i] = np.nan
                        else:
                            result[i] = np.log(a[i] / a[j - 1])
                    else:
                        result[i] = np.nan

            else:
                pass

    return result


def log_return(timestamps, a, diff_time):
    if hasattr(timestamps, "values"):
        timestamps = timestamps.values
    if hasattr(a, "values"):
        a = a.values

    return -_log_return(timestamps, a, -diff_time * 10**9)


def convert_nano(data):
    dates = data[:, 0]
    times = data[:, 1]
    date_time = dates * 1e6 + times
    nanoseconds = pd.to_datetime(date_time, format="%Y%m%d%H%M%S").values.astype(
        np.int64
    )
    return nanoseconds
