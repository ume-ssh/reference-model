# router.py

from fastapi import APIRouter, Depends, Request, status

from ..schema import ErrorResponse, RepeaterRequest, ScoringResponse
from ..scoring.prediction import Prediction
from ..scoring.preprocessing import Preprocessing

# ルーター定義
router = APIRouter()


def get_preprocessor(request: Request) -> Preprocessing:
    """前処理クラスのインスタンス取得用ヘルパー

    Parameters
    ----------
    request : Request
        _description_

    Returns
    ----------
    Preprocessing
        前処理クラスのインスタンス
    """

    return request.app.state.repeater_preprocessor


def get_predictor(request: Request) -> Prediction:
    """推論処理クラスのインスタンス取得用ヘルパー

    Parameters
    ----------
        request : Request
            _description_

    Returns
    -------
        Prediction
            推論処理クラスのインスタンス
    """

    return request.app.state.repeater_predictor


@router.post(
    path="/repeater-pyscore",
    summary="途上スコアリング",
    description="利用回数(全店舗)1回以上の申込に対して与信スコアリングを走らせる",
    status_code=status.HTTP_200_OK,
    response_description="スコアリング結果",
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def repeater_scoring(
    application: RepeaterRequest,
    preprocessor: Preprocessing = Depends(get_preprocessor),
    predictor: Prediction = Depends(get_predictor),
) -> ScoringResponse:
    """途上スコアリング

    Parameters
    ----------
        application : RepeaterRequest
            RepeaterRequestでインスタンス化された申込レコード
        preprocessor : Preprocessing, optional
            前処理クラスのインスタンス
        predictor : Prediction, optional
            推論処理クラスのインスタンス

    Returns
    ----------
        ScoringResult
            スコアリング結果
    """

    # 前処理
    preprocessed_record = preprocessor.main(application.model_dump())  # 辞書型に変換したレコードを引数に渡す
    # 予測
    result = predictor.main(preprocessed_record)

    return ScoringResponse(**result)
