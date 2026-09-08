# data_preprocessing.py

import inspect
from datetime import date, datetime
from typing import Dict, Optional

import numpy as np


class Preprocessing:
    def __init__(self, sex_mapping: Dict, occupation_mapping: Dict, scoring_mode: str) -> None:
        self.sex_mapping = sex_mapping
        self.occupation_mapping = occupation_mapping
        self.scoring_mode = scoring_mode

    def main(self, raw_record: Dict) -> Dict:
        """申込レコードから必要なスコアアイテムを作成する

        Parameters
        ----------
        raw_record : Dict
            リクエストで受け取った申込レコード

        Returns
        -------
        Dict
            前処理済みのレコード
        """

        try:
            # 年齢の算出
            age_dict = self.calculate_age(raw_record["application_datetime"], raw_record["birth_date"])
            # 性別のエンコーディング
            sex_encoded_dict = self.encode_categorical_values(self.sex_mapping, raw_record["sex"])
            # 職業のエンコーディング
            occupation_encoded_dict = self.encode_categorical_values(self.occupation_mapping, raw_record["occupation"])

            if self.scoring_mode == "repeater":
                # 途上スコア用スコアアイテムの作成
                calculated_feature_dict = self.calculate_repeater_items(
                    raw_record["charge_times"],
                    raw_record["ng_times"],
                    raw_record["repayment_times"],
                    raw_record["delay_times"],
                    raw_record["charge_total_fee"],
                    raw_record["last_delay_days"],
                )
                # 途上スコア用前処理済みレコードの辞書作成
                preprocessed_record = (
                    raw_record | age_dict | sex_encoded_dict | occupation_encoded_dict | calculated_feature_dict
                )
            else:
                # 新規スコア用前処理済みレコードの辞書作成
                preprocessed_record = raw_record | age_dict | sex_encoded_dict | occupation_encoded_dict
        except KeyError as e:
            error_loc = error_loc = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
            error_msg = f"{e.__class__.__name__}: {str(e)} is missing."
            return error_loc, error_msg
        except Exception as e:
            error_loc = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
            error_msg = f"{e.__class__.__name__}: {str(e)}"
            return error_loc, error_msg
        else:
            # 正常時は前処理済みレコードを返す
            return preprocessed_record

    def calculate_age(self, application_datetime: datetime, birth_date: date) -> Dict:
        """年齢を算出

        Parameters
        ----------
        application_datetime: datetime
            実行日時
        birth_date: date
            生年月日

        Returns
        ----------
        Dict
            {'age': 年齢}
        """

        try:
            # 実行年月日の日付のみを取得
            application_date = application_datetime.date()
            # 年齢を算出
            age = (application_date - birth_date).days // 365
        except Exception as e:
            # エラー発生時
            error_loc = error_loc = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
            error_msg = f"{e.__class__.__name__}: {str(e)}"
            return error_loc, error_msg
        else:
            # 正常時は年齢の辞書を返す
            return {"age": age}

    def encode_categorical_values(self, mapping: Dict, item: str) -> Dict:
        """入力された項目のカテゴリに応じてOne-Hot Encodingする。

        Parameters
        ----------
        mapping: Dict
            ある項目のマッピング辞書。{'カテゴリ': 'スコアアイテム名'}
        item: str
            Encoding対象の項目

        Returns
        ----------
        Dict
            ある項目をEncodingした結果。{'スコアアイテム名': 0|1}
        """

        return {flag_name: 1 if item == category else 0 for category, flag_name in mapping.items()}

    def calculate_repeater_items(
        self,
        charge_times: int,
        ng_times: int,
        repayment_times: int,
        delay_times: int,
        charge_total_fee: int,
        last_delay_days: Optional[int],
    ) -> Dict:
        """途上用のスコアアイテム算出

        Parameters
        ----------
        charge_times : int
            利用回数(全店舗)
        ng_times : int
            審査NG回数(全店舗)
        repayment_times : int
            返済回数(全店舗)
        delay_times : int
            延滞回数(全店舗)
        charge_total_fee : int
            利用金額(全店舗)
        last_delay_days : Optional[int]
            最終支払い延滞日数

        Returns
        -------
        Dict
            算出したスコアアイテムの辞書
        """

        calculated_feature_dic = {}
        try:
            # 返済回数÷延滞回数(全店舗)
            calculated_feature_dic["repayment_times/delay_times"] = (repayment_times + 1) / (delay_times + 1)
            # 審査NG回数÷利用回数(全店舗)
            calculated_feature_dic["ng_times/charge_times"] = ng_times / charge_times
            # 返済回数比率(全店舗)
            calculated_feature_dic["repayment_ratio"] = repayment_times / charge_times
            # 利用単価(全店舗)
            calculated_feature_dic["charge_unit_price"] = charge_total_fee / charge_times
            # 直近利用の延滞日数
            calculated_feature_dic["last_delay_days"] = (
                np.nan if (last_delay_days is None) or (last_delay_days < 0) else last_delay_days
            )
        except ZeroDivisionError as e:
            # charge_times=0で割り切れない場合
            error_loc = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
            error_msg = "charge_times is 0"
            return error_loc, error_msg
        except Exception as e:
            # その他エラー発生時
            error_loc = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
            return error_loc, f"{e.__class__.__name__}: {str(e)}"
        else:
            # 正常時は算出したスコアアイテムを返す
            return calculated_feature_dic
