import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi import status
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from tqdm import tqdm

from AGPS_pyscore.main import app
from AGPS_pyscore.schema import ErrorResponse, RepeaterRequest, ScoringResponse
from AGPS_pyscore.scoring.prediction import Prediction
from AGPS_pyscore.scoring.preprocessing import Preprocessing


@pytest.fixture()
def client():
    # lifespanを嚙ませる場合は、with文でTestClientを使用して起動前後の処理を走らせる。
    with TestClient(app) as client:
        yield client


def test_get_app(client):
    """APIが生きているかを確認するテスト"""

    # FastAPI起動時のメッセージ返却用
    @app.get("/")
    async def Hello():
        return {"message": "Hello!"}

    response = client.get("/")
    assert response.status_code == 200
    print(response.json())
    assert response.json() == {"message": "Hello!"}
    print("done...")


def test_read_state(client):
    """インスタンスが正常に起動しているかを確認するテスト"""

    # TODO: 新規スコア用インスタンス
    # print("===== 新規スコア =====")

    # 途上スコア用インスタンス
    print("===== 途上スコア =====")
    repeater_preprocessor = client.app.state.repeater_preprocessor
    repeater_predictor = client.app.state.repeater_predictor
    # 検証
    print(repeater_preprocessor)
    print(repeater_predictor)
    assert isinstance(repeater_preprocessor, Preprocessing)
    assert isinstance(repeater_predictor, Prediction)
    print("done...")


class TestPost3A:
    def test_post_new_pyscore(self, client):
        """新規スコアのAPIにリクエストしたときに404Errorが返ってくるかを確認するテスト"""

        record = {
            "id": "string",
            "application_datetime": "2026-01-22T08:39:45.075Z",
            "birth_date": "2026-01-22",
            "sex": "string",
            "occupation": "string",
            "kyc": 0,
            "account_elapsed_hours": 0,
            "phone_missing_days": 0,
            "charge_times": 0,
        }

        # 実行
        response = client.post("/new-pyscore", json=record, headers={"Content-Type": "application/json"})
        print(response.text)
        assert response.status_code == 404

    def test_post_repeater_pyscore(self, client, repeater_data, repeater_results):
        print("===== 途上スコア =====")
        # 実行
        start_time = time.perf_counter()  # 計測開始
        response = client.post("/repeater-pyscore", json=repeater_data[0], headers={"Content-Type": "application/json"})
        end_time = time.perf_counter()  # 計測終了

        # 検証
        assert response.status_code == 200
        for test_val, correct_val in zip(response.json().values(), repeater_results[0].values()):
            print(test_val, correct_val)
            if isinstance(test_val, str) is False:
                assert np.isclose(test_val, correct_val)

        # 処理時間計算
        elapsed_time_ms = (end_time - start_time) * 1000  # ミリ秒
        print(f"time: {elapsed_time_ms:.3f}ms")


class TestPostAllRecord:

    def test_post_repeater_pyscore(self, client, repeater_data, repeater_results):
        elapsed_time_dic = {}
        score_results_list = []

        print("===== 途上スコア =====")
        for record, expected_result in tqdm(zip(repeater_data, repeater_results)):
            # 実行
            start_time = time.perf_counter()  # 計測開始
            response = client.post("/repeater-pyscore", json=record, headers={"Content-Type": "application/json"})
            end_time = time.perf_counter()  # 計測終了
            # 検証
            assert response.status_code == 200
            for test_val, correct_val in zip(response.json().values(), expected_result.values()):
                if isinstance(test_val, str) is False:
                    assert np.isclose(test_val, correct_val)

            # 処理時間計算
            elapsed_time_dic[expected_result["id"]] = (end_time - start_time) * 1000
        print(f"-> time: {np.mean(list(elapsed_time_dic.values())):.3f}ms")
        # pd.Series(elapsed_time_dic).to_csv(Path().cwd() / "data/repeater_api_times.csv")
        del elapsed_time_dic, score_results_list

    @pytest.fixture()
    def application_data(self):
        with open(Path("data/202601_delay_7days_rcscore.json"), "r") as f:
            application_data = json.load(f)

        return application_data

    def test_post_repeater_using_application_data(self, client, application_data):
        score_results_list = []

        print("===== 途上スコア =====")
        for record in tqdm(application_data):
            # 実行
            response = client.post("/repeater-pyscore", json=record, headers={"Content-Type": "application/json"})
            assert response.status_code == 200

            test_result = response.json()
            score_results_list.append([test_result["id"], test_result["score"], test_result["rank"]])

        pd.DataFrame(score_results_list, columns=["ID", "予測未収率_検証", "ランク_検証"]).to_csv(
            Path().cwd() / "data/202601_delay_7days_rcscore検証結果.csv", encoding="cp932"
        )
        del score_results_list


class TestValidationExceptions:
    """RequestValidationError | ResponseValidationErrorが適切に拾われていることを確認するテスト"""

    @pytest.fixture()
    def records(self):
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
        }

    def test_invalid_id(self, client, records):
        """IDに不備があった場合"""

        # テストデータ作成
        repeater_record = self.repeater_record
        repeater_record["id"] = 0
        # 実行
        response = client.post("/repeater-pyscore", json=repeater_record, headers={"Content-Type": "application/json"})
        # 検証
        print(response.text)
        assert response.status_code == 422

    def test_validate_charge_times(self, client, records):
        """charge_timesがschema.pyで設定した値と異なる場合"""

        # テストデータ作成
        repeater_record = self.repeater_record
        repeater_record["charge_times"] = 0
        # 実行
        response = client.post("/repeater-pyscore", json=repeater_record, headers={"Content-Type": "application/json"})
        # 検証
        print(response.text)
        assert response.status_code == 422
        print("done...")

    def test_missing_item(self, client, records):
        """schema.pyで設定したアイテムのいずれかが含まれていなかった場合"""

        # テストデータ作成
        repeater_record = self.repeater_record
        del repeater_record["account_elapsed_hours"], repeater_record["ng_times"]
        print("\n===== 途上スコア =====")
        # 実行
        response = client.post("/repeater-pyscore", json=repeater_record, headers={"Content-Type": "application/json"})
        # 検証
        print(response.text)
        assert response.status_code == 422
        print("done...")

    def test_difference_type(self, client, records):
        """schema.pyで設定した型と異なる型が入力された場合"""

        # テストデータ作成
        repeater_record = self.repeater_record
        repeater_record["birth_date"] = "1983/03/09"
        repeater_record["kyc"] = "True"
        # 実行
        response = client.post("/repeater-pyscore", json=repeater_record, headers={"Content-Type": "application/json"})
        # 検証
        print(response.text)
        assert response.status_code == 422

    def test_response_validationerror(self, client, records):
        """Responseがshcema.pyで設定した型と異なる場合"""

        # スコアリング結果がValidationErrorの場合
        print("===== ScoringResponse =====")

        @app.post("/scoring-response-error")
        async def ScoringResponseError(application: RepeaterRequest) -> ScoringResponse:
            return {"id": 0, "score": 0.000, "rank": 0}

        # 実行
        response = client.post(
            "/scoring-response-error", json=self.repeater_record, headers={"Content-Type": "application/json"}
        )
        print(response.text)
        assert response.status_code == 422

        # エラー結果がValidationErrorの場合
        print("===== ErrorResponse =====")

        @app.post("/error-response-error")
        async def ErrorResponseError(application: RepeaterRequest) -> JSONResponse:
            return JSONResponse(
                content=ErrorResponse(**{"id": 0, "error_loc": "loc", "error_msg": "error"}).model_dump(),
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        # 実行
        response = client.post(
            "/error-response-error", json=self.repeater_record, headers={"Content-Type": "application/json"}
        )
        print(response.text)
        assert response.status_code == 422


def mock_slow_predict(*args, **kwargs):
    time.sleep(10)  # 10秒待機
    return {"id": "id", "score": 0.000, "rank": 10}


class TestExceptions:
    """ValidationError以外の例外処理が拾われていることを確認するテスト"""

    def test_timeouterror(self, client):
        """TimeoutError: レスポンスにx秒以上経っている場合"""

        repeater_record = {
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
        }

        # 推論処理をモックでタイムアウトしてみる
        with patch("AGPS_pyscore.scoring.prediction.Prediction.main", side_effect=mock_slow_predict):
            response = client.post(
                "/repeater-pyscore", json=repeater_record, headers={"Content-Type": "application/json"}
            )
            print(response.text)
            assert response.status_code == 504

    def test_invalid_json(self, client):
        """JSONDecodeError: 正しくない形状のJSONが入力された場合"""

        invalid_record = """
            {
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
            }
        """

        # 実行
        response = client.post("/repeater-pyscore", content=invalid_record, headers={"Contet-Type": "application/json"})
        # 検証
        print(response.text)
        assert response.status_code == 422
