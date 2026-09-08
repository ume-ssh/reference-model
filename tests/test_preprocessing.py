# test_preprocessing.py

import numpy as np
import pytest

from AGPS_pyscore.schema import RepeaterRequest
from AGPS_pyscore.scoring.preprocessing import Preprocessing
from AGPS_pyscore.settings import repeater_settings


@pytest.fixture(scope="package")
def repeater_preprocessing() -> Preprocessing:
    """途上スコア用のインスタンス初期化

    Returns:
        Preprocessing: 途上スコア用前処理インスタンス
    """
    return Preprocessing(repeater_settings.sex_mapping, repeater_settings.occupation_mapping, "repeater")


# @pytest.fixture(scope='package')
# def new_preprocessing():
#     return Preprocessing(new_sex_mapping, new_occupation_mapping, 'new')


class TestPreprocessing3A:
    """単一レコードを用いた基本的な動作確認"""

    @pytest.fixture()
    def return_record(self, repeater_data, preprocessed_repeater_data):
        # テスト用データに対してデータバリデーションを実行
        repeater_record = RepeaterRequest(**repeater_data[1]).__dict__
        # 正しい前処理済みデータに対して、None->nanに変換
        preprocessed_repeater_record = {
            key: np.nan if val is None else val for key, val in preprocessed_repeater_data[0].items()
        }

        return repeater_record, preprocessed_repeater_record

    def test_main(self, repeater_preprocessing, return_record):
        # レコード取得
        repeater_record, correct_preprocessed_repeater_record = return_record

        print("\n ===== 途上スコア =====")
        # 実行
        preprocessed_repeater_record = repeater_preprocessing.main(repeater_record)
        # 検証
        for key, correct_val in correct_preprocessed_repeater_record.items():
            test_val = preprocessed_repeater_record[key]
            print(key, test_val, correct_val)
            if isinstance(test_val) is float or isinstance(test_val) is int:
                assert np.isclose(test_val, correct_val, equal_nan=True)
        print("done...")

    def test_calculate_age(self, repeater_preprocessing, return_record):
        # レコード取得
        repeater_record, correct_preprocessed_repeater_record = return_record

        print("\n ===== 途上スコア =====")
        age_dict = repeater_preprocessing.calculate_age(
            repeater_record["application_datetime"], repeater_record["birth_date"]
        )
        print(age_dict["age"], correct_preprocessed_repeater_record["age"])
        assert age_dict["age"] == correct_preprocessed_repeater_record["age"]
        print("done...")

    def test_encode_sex_vals(self, repeater_preprocessing, return_record):
        """性別のエンコーディングテスト

        Args:
            init_instance (_type_): _description_
            return_record (_type_): _description_
        """

        # レコード取得
        repeater_record, correct_preprocessed_repeater_record = return_record

        # 実行
        print("\n ===== 途上スコア =====")
        sex_encoded_dic = repeater_preprocessing.encode_categorical_values(
            repeater_preprocessing.sex_mapping, repeater_record["sex"]
        )
        # 検証
        print(sex_encoded_dic["sex_male_flag"], correct_preprocessed_repeater_record["sex_male_flag"])
        assert sex_encoded_dic["sex_male_flag"] == correct_preprocessed_repeater_record["sex_male_flag"]

    def test_encode_occupation_vals(self, repeater_preprocessing, return_record):
        """職業のエンコーディングテスト

        Args:
            init_instance (_type_): _description_
            return_record (_type_): _description_
        """
        # レコード取得
        repeater_record, correct_preprocessed_repeater_record = return_record

        print("\n ===== 途上スコア =====")
        # 実行
        occupation_encoded_dic = repeater_preprocessing.encode_categorical_values(
            repeater_preprocessing.occupation_mapping, repeater_record["occupation"]
        )
        correct_occupation_encoded_list = [correct_preprocessed_repeater_record[key] for key in occupation_encoded_dic]
        # 検証
        print(occupation_encoded_dic.values())
        # print(occupation_encoded_dic.values(), correct_occupation_encoded_list)
        # assert list(occupation_encoded_dic.values()) == correct_occupation_encoded_list
        print("done...")

    def test_calculate_repeater_items(self, repeater_preprocessing, return_record):
        # レコード取得
        repeater_record, correct_preprocessed_repeater_record = return_record

        print("\n ===== 途上スコア =====")
        # 実行
        calculated_feature_dic = repeater_preprocessing.calculate_repeater_items(repeater_record)
        # 値をfloat32に変換
        calculated_feature_arr = np.array(list(calculated_feature_dic.values()))
        correct_calculated_feature_arr = np.array(
            [correct_preprocessed_repeater_record[key] for key in calculated_feature_dic]
        )
        # 検証
        print(calculated_feature_arr, correct_calculated_feature_arr)
        assert np.allclose(calculated_feature_arr, correct_calculated_feature_arr, equal_nan=True)
        print("done...")


class TestPreprocessingAllRecord:
    def test_main(self, repeater_preprocessing, repeater_data, preprocessed_repeater_data):
        print("\n===== 途上スコア =====")
        for repeater_record, repeater_expected in zip(repeater_data, preprocessed_repeater_data):
            # 実行
            preprocessed_repeater_record = repeater_preprocessing.main(RepeaterRequest(**repeater_record).__dict__)
            # 検証
            for key, correct_val in repeater_expected.items():
                test_val = preprocessed_repeater_record[key]
                correct_val = np.nan if correct_val is None else correct_val
                if isinstance(test_val) is float or isinstance(test_val) is int:
                    assert np.isclose(test_val, correct_val, equal_nan=True)
        print("done...")

    def test_calculate_repeater_items(self, repeater_preprocessing, repeater_data, preprocessed_repeater_data):
        for repeater_record, repeater_expected in zip(repeater_data, preprocessed_repeater_data):
            # 実行
            calculated_feature_dic = repeater_preprocessing.calculate_repeater_items(repeater_record)
            # 値をfloat32に変換
            calculated_feature_arr = np.array(list(calculated_feature_dic.values()))
            expected_feature_arr = np.array(
                [np.nan if repeater_expected[key] is None else repeater_expected[key] for key in calculated_feature_dic]
            )
            # 検証
            assert np.allclose(calculated_feature_arr, expected_feature_arr, equal_nan=True)


class TestPreprocessinException:
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
        }

    def test_main(self, repeater_preprocessing, record):

        # 年齢算出に必要な項目が無い場合
        print("\n===== 年齢算出 =====")
        missing_record = self.repeater_record.copy()
        del missing_record["birth_date"]
        e = repeater_preprocessing.main(missing_record)
        print(e)
        assert e[0] == "Preprocessing.main"

        # エンコーディングに必要な項目が無い場合
        print("===== 性別エンコーディング =====")
        missing_record = self.repeater_record.copy()
        del missing_record["sex"]
        e = repeater_preprocessing.main(missing_record)
        print(e)
        assert e[0] == "Preprocessing.main"

    def test_calculate_age(self, repeater_preprocessing, record):
        # 年齢算出に必要な項目がdate型でない場合
        e = repeater_preprocessing.calculate_age(
            self.repeater_record["application_datetime"], self.repeater_record["birth_date"]
        )
        print(e)
        assert e[0] == "Preprocessing.calculate_age"

    def test_calculate_repeater_items(self, repeater_preprocessing, record):
        invalid_record = self.repeater_record.copy()
        invalid_record["charge_times"] = 0

        # charge_times=0の時にZeroDivisionErrorが発生するか
        e = repeater_preprocessing.calculate_repeater_items(
            invalid_record["charge_times"],
            invalid_record["ng_times"],
            invalid_record["repayment_times"],
            invalid_record["delay_times"],
            invalid_record["charge_total_fee"],
            invalid_record["last_delay_days"],
        )
        print(e)
        assert e[1] == "charge_times is 0"
