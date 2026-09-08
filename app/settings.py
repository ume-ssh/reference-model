# config.py

from pathlib import Path
from typing import Dict, List

from pydantic_settings import BaseSettings


class PyscoreSettings(BaseSettings):
    # スレッド数
    intra_op_num_threads: int = 1
    # タイムアウト秒数
    timeout: float = 300 / 1000


class RepeaterPyscoreSettings(BaseSettings):
    # 性別用マッピング辞書
    sex_mapping: Dict = {
        "GENDER_MALE": "sex_male_flag",
    }

    # 職業用マッピング辞書
    occupation_mapping: Dict = {
        "OCCUPATION_EMPLOYEE": "occupation_employee_flag",
        "OCCUPATION_HOMEMAKER": "occupation_homemaker_flag",
        "OCCUPATION_PART_TIME_WORKER": "occupation_parttime_flag",
        "OCCUPATION_PROFESSIONAL_JOB": "occupation_professional_flag",
        "OCCUPATION_PUBLIC_OFFICER": "occupation_public_flag",
        "OCCUPATION_SELF_EMPLOYED_WORKER": "occupation_selfworker_flag",
        "OCCUPATION_STUDENT": "occupation_student_flag",
    }

    # モデルパス
    onnx_path: Path = Path("models/repeater_model_v2.onnx")

    # スコアアイテム一覧
    feature_name_list: List[str] = [
        "kyc",
        "account_elapsed_hours",
        "ng_times",
        "charge_times",
        "charge_total_fee",
        "using_fee",
        "repayment_times",
        "repayment_total_fee",
        "repayment_times/delay_times",
        "ng_times/charge_times",
        "sex_male_flag",
        "occupation_employee_flag",
        "occupation_homemaker_flag",
        "occupation_parttime_flag",
        "occupation_professional_flag",
        "occupation_public_flag",
        "occupation_selfworker_flag",
        "occupation_student_flag",
        "last_delay_days",
        "age",
        "repayment_ratio",
        "charge_unit_price",
    ]

    # 閾値一覧(0->10ランク)
    ths_list: List[float] = [
        0.243022,
        0.196452736854553,
        0.106877245008945,
        0.0819738283753395,
        0.0763294845819473,
        0.0660207346081733,
        0.0580369755625724,
        0.0520968027412891,
        0.0334860943257808,
        0.027997376397252,
    ]


repeater_settings = RepeaterPyscoreSettings()
pyscore_settings = PyscoreSettings()
