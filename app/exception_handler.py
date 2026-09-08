import asyncio
from json.decoder import JSONDecodeError
from typing import Union

from anyio import fail_after
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from pydantic_core import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

from .schema import ErrorResponse


class ExceptionMiddlware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, timeout: float):
        super().__init__(app)
        self.timeout = float(timeout)  # タイムアウト秒数

    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            if request.method == "POST":
                record = await request.json()
            with fail_after(self.timeout):
                response = await call_next(request)
                return response
        except JSONDecodeError as exc:
            # リクエストされたJSONに不備があった場合
            error_content = {
                "id": "unknown",
                "error_loc": f"{request.method} to {request.url}",
                "error_msg": f"JSONDecodeError: {str(exc)}",
            }
            return JSONResponse(
                content=ErrorResponse(**error_content).model_dump(),
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )
        except TimeoutError:
            # タイムアウトエラー
            error_content = {
                "id": record["id"],
                "error_loc": f"{request.method} to {request.url}",
                "error_msg": f"Request Processing time excedeed {self.timeout/1000}(ms)",
            }
            return JSONResponse(
                content=ErrorResponse(**error_content).model_dump(), status_code=status.HTTP_504_GATEWAY_TIMEOUT
            )
        except Exception as exc:
            # その他エラー発生時
            error_loc = exc[0]
            error_msg = exc[1]
            error_content = {"id": record["id"], "error_loc": error_loc, "error_msg": error_msg}
            return JSONResponse(
                content=ErrorResponse(**error_content).model_dump(),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


async def validation_exception_handler(
    request: Request, exc: Union[RequestValidationError, ResponseValidationError, ValidationError]
) -> JSONResponse:
    """リクエストorレスポンスにpydanticのValidationErrorが発生した場合"""

    # 申込レコードの中身を取得
    record = await request.json()

    # IDに不備があったときはunknownで返す
    if "id" in record:
        id = record["id"]
    else:
        id = "unknown"

    # エラーが発生した項目ごとにエラーメッセージを格納する
    if exc.__class__.__name__ == "RequestValidationError":
        error_loc = f"Request to {request.url}"
        error_msg = {msg["type"]: f"{msg['loc'][1]} is {msg['msg']}" for msg in exc.errors()}
    elif exc.__class__.__name__ == "ResponseValidationError":
        error_loc = f"Response from {request.url}"
        error_msg = {msg["type"]: f"{msg['loc'][1]} is {msg['msg']}" for msg in exc.errors()}
    else:
        error_loc = f"Error from {request.url}"
        error_msg = {msg["type"]: f"{msg['loc']} is {msg['msg']}" for msg in exc.errors()}

    # レスポンス用辞書
    error_content = {"id": id, "error_loc": error_loc, "error_msg": error_msg}

    return JSONResponse(
        content=ErrorResponse(**error_content).model_dump(), status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )
