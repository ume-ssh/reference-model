# schema.py

from datetime import date, datetime
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field


# 新規申込レコード用データバリデーション
class NewRequest(BaseModel):
    id: int
    application_datetime: datetime
    birth_date: date
    sex: str
    occupation: str
    kyc: int
    account_elapsed_hours: float
    phone_missing_days: int
    charge_times: int


# 途上申込レコード用データバリデーション
class RepeaterRequest(BaseModel):
    id: int = Field(description="レコードに紐づくID")
    application_datetime: datetime = Field(description="実行日時")
    birth_date: date = Field(description="生年月日")
    sex: str = Field(description="性別")
    occupation: Optional[str] = Field(description="職業")
    kyc: int = Field(description="認証済み")
    ng_times: int = Field(description="審査NG回数(全店舗)")
    charge_times: int = Field(gt=0, description="利用回数(全店舗)")
    charge_total_fee: int = Field(description="利用金額(全店舗)")
    using_fee: int = Field(description="利用中合計金額(全店舗)")
    repayment_times: int = Field(description="返済回数(全店舗)")
    repayment_total_fee: int = Field(description="返済金額(全店舗)")
    delay_times: int = Field(description="延滞回数(全店舗)")
    last_delay_days: Optional[int] = Field(description="最終支払い延滞日数")
    account_elapsed_hours: float = Field(description="アカウント経過時間(時)")


# スコアリング結果用データバリデーション
class ScoringResponse(BaseModel):
    id: int = Field(description="リクエストに紐づくID")
    score: float = Field(description="予測未収率")
    rank: int = Field(description="ランク")


# エラー結果用データバリデーション
class ErrorResponse(BaseModel):
    id: Any = Field(description="リクエストに紐づくID")
    error_loc: str = Field(description="エラー発生箇所")
    error_msg: Union[str, Dict[str, str]] = Field(description="エラー内容")
