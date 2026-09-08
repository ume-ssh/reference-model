from fastapi import APIRouter, status

from ..schema import ErrorResponse, NewRequest, ScoringResponse

# ルーター定義
router = APIRouter()


@router.post(
    path="/new-pyscore",
    summary="新規スコア",
    description="利用回数(全店舗)0回の申込に対して走らせる与信スコア",
    status_code=status.HTTP_400_BAD_REQUEST,
    response_description="400エラーメッセージ",
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def new_scoring(application: NewRequest) -> ErrorResponse:
    """新規スコア実装までは400エラーを返す

    Parameters
    ----------
        application : NewRequest
            NewRequestでインスタンス化された申込レコード

    Returns
    -------
        ErrorResponse
            400エラーメッセージ
    """
    # TODO: 新規スコア導入の目途が立ってから実装

    error_content = {
        "id": application.id,
        "error_loc": "POST to /new-pscore",
        "error_msg": "new-pyscore not yet run",
    }

    return ErrorResponse(**error_content)
