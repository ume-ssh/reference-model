import json
import pickle
import time
from pathlib import Path

import pytest


@pytest.fixture(scope="package")
def repeater_data():
    with open(Path("data/repeater_test_data.json"), "r") as f:
        repeater_data = json.load(f)

    return repeater_data


@pytest.fixture(scope="package")
def preprocessed_repeater_data():
    with open(Path("data/repeater_preprocessed_test_data.json"), "r") as f:
        preprocessed_repeater_data = json.load(f)

    return preprocessed_repeater_data


@pytest.fixture(scope="function")
def repeater_X():
    with open(Path("data/repeater_test_X_arr.pkl"), "rb") as f:
        repeater_X = pickle.load(f)

    return repeater_X


@pytest.fixture(scope="function")
def repeater_results():
    with open(Path("data/repeater_test_results_2threads_v2.pkl"), "rb") as f:
        repeater_results = pickle.load(f)

    return repeater_results


@pytest.fixture()
def measure_elapsed_times():
    start_time = time.perf_counter()  # 計測開始
    yield
    end_time = time.perf_counter()  # 計測終了
    elapsed_time_ms = (end_time - start_time) * 1000  # ミリ秒で処理時間を算出
    print(f"処理時間: {elapsed_time_ms:.3f}ms")
