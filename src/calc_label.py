import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import warnings
import concurrent.futures
import sutils
import common as cm  # 确保 common 模块可用

warnings.filterwarnings("ignore")


def calculate_stats(df, current, future):
    log_returns = np.log(df["mid_price"].shift(-60) / df["mid_price"]) * 1e4
    df[f"mean_{current}-{future}"] = (
        np.log(
            (df["mid_price"].rolling(future - current).mean().shift(-future))
            / df["mid_price"]
        )
        * 1e4
    )
    df[f"var_{current}-{future}"] = (
        log_returns.rolling(future - current).var().shift(-future)
    )
    df[f"vol_{current}-{future}"] = (
        log_returns.rolling(future - current).std().shift(-future)
    )


def calculate_stats2(df, current, future):
    df[f"max_{current}-{future}"] = (
        np.log(
            (df["mid_price"].rolling(future - current).max().shift(-future))
            / df["mid_price"]
        )
        * 1e4
    )
    df[f"min_{current}-{future}"] = (
        np.log(
            df["mid_price"].rolling(future - current).min().shift(-future)
            / df["mid_price"]
        )
        * 1e4
    )
    df[f"gap_{current}-{future}"] = (
        (
            df["mid_price"].rolling(future - current).max().shift(-future)
            - df["mid_price"].rolling(future - current).min().shift(-future)
        )
        / df["mid_price"]
        * 1e4
    )


def calculate_stats3(df, cur, future):
    results = []
    for name, group in df.groupby(df.index):
        future_prices = group["mid_price"].shift(-future)
        last_valid_price = group["mid_price"].iloc[-1]
        future_prices.fillna(last_valid_price, inplace=True)
        ret = np.log(future_prices / group["mid_price"]) * 1e4
        ret = pd.DataFrame(ret, index=group.index, columns=[f"ret_{future}"])
        results.append(ret)
    return pd.concat(results)


def process_code(code, use_multithreading):
    datas = cm.get_snapshot(code)
    df = datas["tickData"]
    tm = datas["timestamp"][:, 0]
    mask = tm < 20240101
    df = df[mask]
    tm = tm[mask]
    df = pd.DataFrame(df, columns=cm.COLS_SNAPSHOTS)
    df.index = tm

    original_columns = set(df.columns)
    df["mid_price"] = (df["AskPrice1"] + df["BidPrice1"]) / 2
    cur_futs = [(0, 60), (60, 120), (60, 300), (60, 600), (120, 180)]
    for cur, futs in cur_futs:
        calculate_stats(df, cur, futs)
        calculate_stats2(df, cur, futs)
        print(f"calculated {cur}-{futs}")
        # df[f"ret_{futs}"] = calculate_stats3(df, cur, futs)

    new_columns = set(df.columns) - original_columns
    df = df.fillna(0)

    save_path = f"/mnt/nas/data/WY/label_1000/{code}/"
    os.makedirs(save_path, exist_ok=True)
    for col in new_columns:
        feature_path = os.path.join(save_path, f"{col}.npy")
        np.save(feature_path, df[col].values.reshape(-1, 1).astype(np.float32))
        print(f"Saved {feature_path}")


def main(use_multithreading=True):
    stk_list = [code for code in cm.STK_CODES if code not in cm.SELECTED_CODES]

    if use_multithreading:
        print("Using multithreading")
        with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(process_code, code, use_multithreading)
                for code in stk_list
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(stk_list)
            ):
                future.result()  # 处理异常
    else:
        print("Not using multithreading")
        for code in tqdm(reversed(stk_list)):
            process_code(code, use_multithreading)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process stock data with optional multithreading."
    )
    parser.add_argument(
        "--multithreading",
        "-m",
        default=False,
    )
    args = parser.parse_args()

    main(use_multithreading=True)
