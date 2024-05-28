import h5py
import pandas as pd
import numpy as np
from glob import glob
import os
import time
import bottleneck as bn
import numba as nb
from functools import lru_cache
from typing import *
from numba import njit
import torch as th
from sklearn.model_selection import train_test_split

PATH_SNAPSHOTS = r"/mnt/nas/data/data_universe/data_hub"
PATH_FACTORS = r"\\192.168.88.80\lc\股票t0研究\数据\股票数据hdf5"
PATH_WY = r"\\192.168.88.80\lc\股票t0研究\wy"
PATH_FACTORS = r"H:\data\factors"

files = glob(os.path.join(PATH_FACTORS, "*.parquet"))
STK_CODES = [os.path.basename(file).split("_")[1].split(".")[0] for file in files]
dates = [
    20210401,
    20210701,
    20211001,
    20220101,
    20220401,
    20220701,
    20221001,
    20230101,
]
COLS_SNAPSHOTS = [
    "PreClose",
    "Open",
    "High",
    "Low",
    "Price",
    "Volume",
    "Turover",
    "AccVolume",
    "AccTurover",
    "MatchItems",
    "TotalBidVolume",
    "TotalAskVolume",
    "BidAvPrice",
    "AskAvPrice",
    "BidPrice1",
    "BidPrice2",
    "BidPrice3",
    "BidPrice4",
    "BidPrice5",
    "BidPrice6",
    "BidPrice7",
    "BidPrice8",
    "BidPrice9",
    "BidPrice10",
    "BidVolume1",
    "BidVolume2",
    "BidVolume3",
    "BidVolume4",
    "BidVolume5",
    "BidVolume6",
    "BidVolume7",
    "BidVolume8",
    "BidVolume9",
    "BidVolume10",
    "AskPrice1",
    "AskPrice2",
    "AskPrice3",
    "AskPrice4",
    "AskPrice5",
    "AskPrice6",
    "AskPrice7",
    "AskPrice8",
    "AskPrice9",
    "AskPrice10",
    "AskVolume1",
    "AskVolume2",
    "AskVolume3",
    "AskVolume4",
    "AskVolume5",
    "AskVolume6",
    "AskVolume7",
    "AskVolume8",
    "AskVolume9",
    "AskVolume10",
]


SELECTED_CODES = [
    "000537",
    "000627",
    "000925",
    "000950",
    "002058",
    "002166",
    "002308",
    "002399",
    "002498",
    "002557",
    "002577",
    "002594",
    "002901",
    "002941",
    "002946",
    "300053",
    "300137",
    "300141",
    "300215",
    "300225",
    "300241",
    "300252",
    "300366",
    "300498",
    "300564",
    "300605",
    "300640",
    "300688",
    "300713",
    "300867",
    "300870",
    "300908",
    "300913",
    "600006",
    "600012",
    "600107",
    "600123",
    "600127",
    "600163",
    "600176",
    "600218",
    "600232",
    "600267",
    "600302",
    "600395",
    "600426",
    "600428",
    "600493",
    "600557",
    "600578",
    "600644",
    "600647",
    "600665",
    "600704",
    "600740",
    "600797",
    "600817",
    "600834",
    "600859",
    "600862",
    "600893",
    "600984",
    "601019",
    "601330",
    "601881",
    "603006",
    "603017",
    "603018",
    "603037",
    "603192",
    "603212",
    "603269",
    "603357",
    "603368",
    "603388",
    "603390",
    "603559",
    "603595",
    "603693",
    "603712",
    "603777",
    "603818",
    "603856",
    "603878",
    "603939",
    "603990",
    "605128",
    "605166",
    "688057",
    "688165",
    "688215",
    "688286",
    "688309",
    "688313",
    "688366",
    "688386",
    "688668",
    "688678",
    "688777",
    "689009",
]

COLS_FACTORS_TICKDATA = [
    "MidPrice",
    "BidPrice1",
    "BidVolume1",
    "AskPrice1",
    "AskVolume1",
    "Closes",
]


def print_all_datasets(file_path, prt=False):
    columns = []

    def print_name(name, obj):
        if isinstance(obj, h5py.Dataset):
            if prt:
                print(f"Dataset Path: {name}")
            columns.append(name)

    with h5py.File(file_path, "r") as f:
        # 使用visititems方法遍历文件中的所有对象，并对每个对象调用print_name函数
        f.visititems(print_name)
    return columns


@lru_cache()
def get_snapshot(code: str = "000006"):
    if isinstance(code, int):
        code = f"{code:06d}"
    file_path = os.path.join(PATH_SNAPSHOTS, f"stkCode_{code}.h5")
    f: dict[str, np.ndarray] = h5py.File(file_path, "r")
    cols = print_all_datasets(file_path)
    ret = {f"{col}": f[col][:] for col in cols}
    return ret


@lru_cache()
def get_factor(code: str = "000006") -> Dict:
    file_path = os.path.join(PATH_FACTORS, f"stkCode_{code}.h5")
    f: dict[str, np.ndarray] = h5py.File(file_path, "r")
    cols = print_all_datasets(file_path)
    ret = {f"{col}": f[col][:] for col in cols}
    return ret


@lru_cache()
def get_X(code, start, end):
    """
    Retrieve data for a given stock code within a specified date range,
    avoiding the use of DataFrame to enhance processing speed.

    Parameters:
    - code: The stock code.
    - start: The start date in "YYYYMMDD" format.
    - end: The end date in "YYYYMMDD" format.

    Returns:
    - A dictionary containing the filtered factor data, labels, and timestamps.
    """
    dict_factor = get_factor(code)
    start_date = int(start)
    end_date = int(end)
    date_array = dict_factor["timestamp"][:, 0]
    indices = (date_array >= start_date) & (date_array < end_date)
    filtered_factors = dict_factor["factor"][indices]
    filtered_labels = dict_factor["label"][indices]
    filtered_timestamps = dict_factor["timestamp"][indices]
    filtered_tickData = dict_factor["tickData"][indices]
    result = {
        "factor": filtered_factors,
        "label": filtered_labels,
        "timestamp": filtered_timestamps,
        "tickData": filtered_tickData,
    }

    return result


# @lru_cache(maxsize=10000)
def fetch_dataset(code="600001", train_end="20220601"):
    """
    Fetches datasets for a given stock code, split into a training set ending on 'train_end'
    and spanning one year prior, and a test set spanning three months after 'train_end'.

    Parameters:
    - code: Stock code as a string, default is "600001".
    - train_end: End date for the training dataset in "YYYYMMDD" format.

    Returns:
    - Two dictionaries containing the training and test datasets, respectively.
    """
    file_path = os.path.join(PATH_FACTORS, f"stkCode_{code}.h5")
    with h5py.File(file_path, "r") as f:
        # Assuming 'timestamp' is one of the datasets in the HDF5 file
        timestamps = f["timestamp"][:, 0]

        # Convert 'train_end' and dataset timestamps to pandas datetime for easy manipulation
        train_end_date = pd.to_datetime(train_end, format="%Y%m%d")
        timestamps_date = pd.to_datetime(timestamps.flatten(), format="%Y%m%d")

        # Define start and end dates for the training and test sets
        train_start_date = train_end_date - pd.DateOffset(years=1)
        test_end_date = train_end_date + pd.DateOffset(months=3)

        # Adjust start and end dates based on the available data
        available_start_date = timestamps_date.min()
        available_end_date = min(
            timestamps_date.max(), pd.to_datetime("20230101", format="%Y%m%d")
        )

        if train_start_date < available_start_date:
            train_start_date = available_start_date
        if test_end_date > available_end_date:
            test_end_date = available_end_date
        # Select indices for training and test sets
        train_indices = (timestamps_date >= train_start_date) & (
            timestamps_date < train_end_date
        )
        test_indices = (timestamps_date > train_end_date) & (
            timestamps_date <= test_end_date
        )
        # Fetch datasets for training and test sets
        s = time.time()
        train_set = {f"{col}": f[col][:][train_indices] for col in f.keys()}
        test_set = {f"{col}": f[col][:][test_indices] for col in f.keys()}
        e = time.time()
        print("fetch data elapsed:", int(e - s))

    return train_set, test_set


@njit()
def _cor_calc(x, y):
    n = len(x)
    return (
        np.nansum((x - np.nanmean(x)) * (y - np.nanmean(y)))
        / np.sqrt(np.nanvar(x) * np.nanvar(y))
        / n
    )


@njit()
def _cor_calc_mat(X, y):
    n = len(X)
    X_var = np.ones(X.shape[1]).reshape(-1)
    for i in range(X.shape[1]):
        X_var[i] = np.nanvar(X[:, i])
    X_mean = np.ones(X.shape[1]).reshape(-1)
    for i in range(X.shape[1]):
        X_mean[i] = np.nanmean(X[:, i])
    return (
        (np.dot(X.T, y) - X_mean * np.nanmean(y) * n)
        / np.sqrt(X_var * np.nanvar(y))
        / n
    )


def corr(x, y, mask=None, x_percent=None, y_percent=None):
    """
    :x: list-like. ndim=1/2. x_ndim and y_ndim cannot both be 2
    :y: list-like
    :mask: np.mask, 1 blocked, 0 exposed
    :x_percent: two element list, 0-100. If both x_percent and y_percent are not None, only calculate x_percent
    :y_percent: two element list, 0-100

    return: corr array
    """
    # x_percenty_percent: from 0-100

    if hasattr(x, "values"):
        x = x.values
    if hasattr(y, "values"):
        y = y.values

    if x.shape[0] != y.shape[0]:
        raise Warning("the shape of x and y is not consistent!")
    if x.ndim == 2 and 1 in x.shape:
        x = x.reshape(-1)
    if y.ndim == 2 and 1 in y.shape:
        y = y.reshape(-1)

    if y.ndim != 1 and x.ndim == 1:
        temp = y
        y = x
        x = temp

    if x.ndim == 1 and y.ndim == 1:
        if x_percent is not None:
            mask = np.ones_like(x)
            mask[
                np.where(
                    (x < np.percentile(x, x_percent[0]))
                    | (x > np.percentile(x, x_percent[1]))
                )
            ] = 0
        elif y_percent is not None:
            mask = np.ones_like(y)
            mask[
                np.where(
                    (y < np.percentile(y, y_percent[0]))
                    | (y > np.percentile(y, y_percent[1]))
                )
            ] = 0

    # nan
    if np.isnan(x).any() or np.isnan(y).any():
        temp_df = pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis=1, sort=True)
        temp_df = temp_df.dropna()
        x = temp_df.iloc[:, :-1].values
        y = temp_df.iloc[:, -1].values

    if mask is not None:
        x = x[mask == 0]
        y = y[mask == 0]

    if x.ndim != 1 and y.ndim == 1:
        return _cor_calc_mat(x, y)
    elif x.ndim == 1 and y.ndim == 1:
        return _cor_calc(x, y)
    else:
        return None


def get_stock_board(stock_id):
    """Get stock board from stock ID (symbol)."""
    if stock_id[:2] == "60":
        return "SHM"
    elif stock_id[:3] == "000" or stock_id[:3] == "001" or stock_id[:3] == "003":
        return "SZM"
    elif stock_id[:3] == "300":
        return "GME"  # 创业板
    elif stock_id[:3] == "002":
        return "SME"  # 中小版
    elif stock_id[:3] == "688" or stock_id[:3] == "689":
        return "STAR"  # 科创版
    else:
        return "UNKNOWN"


from rich.progress import track as tqdm


def get_data(code):
    x = np.memmap(
        f"A:/data/factors/factor_{code}.npy", dtype=np.float32, mode="r"
    ).reshape(-1, 101)
    y = np.memmap(f"A:/data/factors/label_{code}.npy", dtype=np.float32, mode="r")
    timestamp = np.memmap(
        f"A:/data/factors/timestamp_{code}.npy", dtype=np.int64, mode="r"
    )
    return x, y, timestamp


def get_label(code):
    means = np.memmap(f"A:/sa/labels/mean_{code}", dtype=np.float32, mode="r")
    means = means.reshape(-1, 3, 3)

    rvs = np.memmap(f"A:/sa/labels/rv_{code}", dtype=np.float32, mode="r")
    rvs = rvs.reshape(-1, 3, 3)

    vars = np.memmap(f"A:/sa/labels/var_{code}", dtype=np.float32, mode="r")
    vars = vars.reshape(-1, 3, 3)
    return means, vars, rvs


def generate_tsdata(data, seq_len):
    N, F = data.shape
    padded_data = th.nn.functional.pad(
        data, (0, 0, seq_len - 1, 0), mode="constant", value=0
    )
    unfold_data = padded_data.unfold(dimension=0, size=seq_len, step=1)
    timeseries_data = unfold_data.permute(0, 2, 1)
    return timeseries_data


def get_mydataset(
    stk_list: List[str],
    fold: int = 0,
    split_type: int = 0,
    seq_len: int = 0,
    seed: int = 2024,
):
    x_trains, x_tests, y_trains, y_tests, x_valids, y_valids = [], [], [], [], [], []
    codes = []
    test_x_dict = {}
    test_y_dict = {}
    train_end = dates[fold]
    test_end = dates[fold + 1]
    for code in tqdm(stk_list):
        df, label, ts = get_data(code)
        train_idx = (ts < train_end) & (ts >= train_end - 10000)
        test_idx = (ts < test_end) & (ts >= train_end)
        x_train, x_test = df[train_idx, :], df[test_idx, :]
        y_train, y_test = label[train_idx].reshape(-1, 1), label[test_idx].reshape(
            -1, 1
        )

        x_train, x_test, y_train, y_test = map(
            lambda x: th.from_numpy(x.copy()),
            (x_train, x_test, y_train, y_test),
        )

        if seq_len > 0:
            x_train, x_test = generate_tsdata(x_train, seq_len), generate_tsdata(
                x_test, seq_len
            )

        idx = list(range(len(y_train)))

        if split_type == 0:
            train_idx, valid_idx = train_test_split(
                idx, test_size=0.3, shuffle=True, random_state=seed
            )

        elif split_type == 1:
            split = int(len(idx) * 0.7)
            train_idx, valid_idx = idx[:split], idx[split:]

        else:
            split = int(len(idx) * 0.7)
            gap = int(len(idx) * 0.01)
            train_idx, valid_idx = idx[:split], idx[split + gap :]

        x_train, x_valid, y_train, y_valid = (
            x_train[train_idx],
            x_train[valid_idx],
            y_train[train_idx],
            y_train[valid_idx],
        )

        x_trains.append(x_train)
        x_valids.append(x_valid)
        x_tests.append(x_test)
        y_trains.append(y_train)
        y_valids.append(y_valid)
        y_tests.append(y_test)
        test_x_dict[code] = x_test
        test_y_dict[code] = y_test

    x_train, x_valid, x_test, y_train, y_valid, y_test = map(
        th.vstack, (x_trains, x_valids, x_tests, y_trains, y_valids, y_tests)
    )
    return (
        x_train,
        x_valid,
        x_test,
        y_train,
        y_valid,
        y_test,
        test_x_dict,
        test_y_dict,
    )


def contribute(a, b, g):
    am = bn.nanmean(a)
    bm = bn.nanmean(b)
    a0 = a - am
    b0 = b - bm
    s2 = np.sqrt(bn.nansum(a0**2) * bn.nansum(b0**2))
    ab = a0 * b0 / s2
    return np.array([ab[g == i].sum() for i in range(10)])


def ic_contribution(df, name, rety="ret", key=["date", "times"]):
    df["rank"] = (
        df.groupby(key)[name]
        .apply(lambda x: pd.qcut(x, 10, labels=range(10)))
        .reset_index(drop=True)
    )
    l = []
    for k, group in df.groupby(key):
        ic = contribute(group[name].values, group[rety].values, group["rank"].values)
        ic[np.isnan(ic)] = 0
        l.append(ic)
    return pd.Series(
        bn.nanmean(np.vstack(l), axis=0), index=[f"c{i:02d}" for i in range(10)]
    )


QRET_INDEX = ["q01", "q05", "q10", "q90", "q95", "q98", "q99"]


def qret(df, name, rety="ret", key=["date", "times"]):
    df["rank"] = df.groupby(key)[name].rank(pct=1, ascending=True)
    q9 = df.loc[df["rank"] > 0.9].groupby(key)[rety].mean().mean()
    q95 = df.loc[df["rank"] > 0.95].groupby(key)[rety].mean().mean()
    q98 = df.loc[df["rank"] > 0.98].groupby(key)[rety].mean().mean()
    q99 = df.loc[df["rank"] > 0.99].groupby(key)[rety].mean().mean()
    q0 = df.loc[df["rank"] < 0.1].groupby(key)[rety].mean().mean()
    q05 = df.loc[df["rank"] < 0.05].groupby(key)[rety].mean().mean()
    q01 = df.loc[df["rank"] < 0.01].groupby(key)[rety].mean().mean()
    return pd.Series(
        [q01, q05, q0, q9, q95, q98, q99],
        index=QRET_INDEX,
    )


def calc_ic(x, y):
    mask = np.isfinite(y)
    y = y[mask]
    x = x[mask]
    return y.dot(x) / np.linalg.norm(x) / np.linalg.norm(y)


def daily_ic(df):
    ic_values = df.groupby("date").apply(
        lambda x: corr(x["yhat"].values, x["y"].values)
    )
    return np.mean(ic_values)


def daily_performance(yhat, y, ts):
    df = pd.DataFrame({"yhat": yhat.flatten(), "y": y.flatten(), "date": ts.flatten()})
    pic = corr(yhat, y)
    bic = daily_ic(df)
    q = qret(df, "yhat", rety="y", key=["date"])
    # icc = ic_contribution(df, "yhat", rety="y", key=["date"])
    return pic, bic, q
