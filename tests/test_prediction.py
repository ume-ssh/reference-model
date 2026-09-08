# test_predicition.py

import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from AGPS_pyscore.scoring.prediction import Prediction
from AGPS_pyscore.settings import repeater_settings


@pytest.fixture(scope="package")
def repeater_predictor() -> Prediction:

    return Prediction(repeater_settings.onnx_path, repeater_settings.feature_name_list, repeater_settings.ths_list)


class TestPrediction3A:
    """単一レコードを用いた基本的な動作確認"""

    @pytest.fixture()
    def return_record(
        self, preprocessed_repeater_data: Dict, repeater_X: np.array, repeater_results: Dict
    ) -> Tuple[Dict, np.array, Dict]:
        """_summary_

        Args:
            preprocessed_repeater_data (Dict): _description_
            repeater_X (np.array): _description_
            repeater_results (Dict): _description_

        Returns:
            preprocessed_repeater_record: 前処理済みレコード
            repeater_x: np.float32型のモデル入力値
            repeater_result: {id, score, rank}のスコアリング結果
        """

        # 前処理済みレコード
        preprocessed_repeater_record = preprocessed_repeater_data[0]
        # スコアアイテム配列
        repeater_x = repeater_X[0].reshape(1, -1)
        # 予測結果
        repeater_result = repeater_results[0]

        return preprocessed_repeater_record, repeater_x, repeater_result

    def test_main(self, repeater_predictor: Prediction, return_record: Dict):
        # 単一レコード取得
        preprocessed_repeater_record, _, correct_repeater_result = return_record

        print("\n ===== 途上スコア =====")
        # 実行
        result = repeater_predictor.main(preprocessed_repeater_record)
        # 検証
        print(result, correct_repeater_result)
        assert result == correct_repeater_result
        # 処理時間
        print("done...")

    def test_make_feature(self, repeater_predictor, return_record):
        preprocessed_repeater_record, correct_repeater_x, _ = return_record

        print("\n ===== 単一レコード =====")
        # 実行
        repeater_x = repeater_predictor.make_features(preprocessed_repeater_record)
        # 検証
        print(repeater_x, correct_repeater_x)
        assert np.array_equal(repeater_x, correct_repeater_x, equal_nan=True)
        print("done...")

    def test_predict(self, repeater_predictor, return_record):
        _, repeater_x, repeater_result = return_record

        print("\n ===== 途上スコア =====")
        # 実行
        y_prob = repeater_predictor.predict(repeater_x)
        # 検証
        print(y_prob, repeater_result["score"])
        assert y_prob == repeater_result["score"]

    def test_output_rank(self, repeater_predictor, return_record):
        _, _, repeater_result = return_record
        # 実行
        rank = repeater_predictor.output_rank(repeater_result["score"])
        # 検証
        print(rank, repeater_result["rank"])
        assert rank == repeater_result["rank"]


class TestPredictionParametrized:
    def test_main(self, repeater_predictor, preprocessed_repeater_data, repeater_results):
        print("\n ===== 途上スコア =====")
        for repeater_record, repeater_result in zip(preprocessed_repeater_data, repeater_results):
            result = repeater_predictor.main(repeater_record)  # 実行
            assert result == repeater_result  # 検証
        print("done...")

    def test_predict(self, repeater_predictor, repeater_X, repeater_results):
        elapsed_time_dict = {}

        print("\n ===== 途上スコア =====")
        for repeater_x, repeater_result in zip(repeater_X, repeater_results):
            # 実行
            start_time = time.perf_counter()  # 開始時間
            y_prob = repeater_predictor.predict(repeater_x.reshape(1, -1))
            end_time = time.perf_counter()  # 終了時間
            # 検証
            assert y_prob == repeater_result["score"]
            elapsed_time_dict[repeater_result["id"]] = (end_time - start_time) * 1000  # 処理時間
        print(f"elapsed_mean_time -> {np.mean(list(elapsed_time_dict.values())):.3f}ms")

        # pd.Series(elapsed_time_dict).to_csv(Path() / "data/repeater_prediction_times.csv")


class TestPredictionException:

    @pytest.fixture()
    def record(self):
        self.repeater_record = {
            "id": "psr:00a8hhbdmaf5hoqf",
            "application_datetime": "2025-05-01T00:00:00",
            "birth_date": "1983-03-09",
            "sex": "GENDER_NOT_ANSWERED",
            "occupation": "OCCUPATION_EMPLOYEE",
            "kyc": 1,
            "ng_times": 2,
            "charge_times": 33,
            "charge_total_fee": 579000,
            "using_fee": 0,
            "repayment_times": 24,
            "repayment_total_fee": 579000,
            "delay_times": 0,
            "last_delay_days": -6.0,
            "account_elapsed_hours": 33891.51953,
            "repayment_times/delay_times": 25,
            "ng_times/charge_times": 0.06,
            "sex_male_flag": 0,
            "occupation_employee_flag": 1,
            "occupation_homemaker_flag": 0,
            "occupation_parttime_flag": 0,
            "occupation_professional_flag": 0,
            "occupation_public_flag": 0,
            "occupation_selfworker_flag": 0,
            "occupation_student_flag": 0,
            "last_delay_days": np.nan,
            "age": 42,
            "repayment_ratio": 0.73,
            "charge_unit_price": 17545,
        }

    def test_main(self, repeater_predictor, record):
        """前処理済みレコードにidが無い場合"""

        del self.repeater_record["id"]
        e = repeater_predictor.main(self.repeater_record)
        print(e)
        assert e[1] == "KeyError: 'id' is missing."

    def test_make_features(self, repeater_predictor, record):
        """前処理済みレコードにいずれかのスコアアイテムが含まれていない場合"""

        del self.repeater_record["charge_unit_price"]
        e = repeater_predictor.make_features(self.repeater_record)
        print(e)
        assert e[1] == "KeyError: 'charge_unit_price' is missing."

    def test_predict(self, repeater_predictor):
        """スコアアイテムの配列が想定と異なる場合"""

        invalid_X = np.random.rand(22, 1).astype(np.float32)
        e = repeater_predictor.predict(invalid_X)
        print(e)
        assert e[0] == "Prediction.predict"
