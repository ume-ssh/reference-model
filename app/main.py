from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .exception_handler import ExceptionMiddlware, validation_exception_handler
from .router import new_pyscore, repeater_pyscore
from .scoring.prediction import Prediction
from .scoring.preprocessing import Preprocessing
from .settings import pyscore_settings, repeater_settings


# インスタンス生成
@asynccontextmanager
async def lifespan(app: FastAPI):
    """API起動時における前処理クラスと推論処理クラスのインスタンス生成

    Parameters
    ----------
        app : FastAPI
    """
    print("----- Startup Pyscore-API -----")

    # 新規スコア
    # TODO: 新規スコア導入の目途が立ってから実装

    # 途上スコア
    app.state.repeater_preprocessor = Preprocessing(
        repeater_settings.sex_mapping, repeater_settings.occupation_mapping, "repeater"
    )
    app.state.repeater_predictor = Prediction(
        repeater_settings.onnx_path, repeater_settings.feature_name_list, repeater_settings.ths_list
    )
    yield

    print("----- Shutdown Pyscore-API -----")


# FastAPI起動用のappを作成
app = FastAPI(
    lifespan=lifespan,
    docs_url=None,
    title="AGPS-pyscore API",
    description="【AGPS後払いチャージ】PythonスコアリングシステムAPIのドキュメント",
    version="0.1.0",
)

# ミドルウェアと例外ハンドラーをappに追加
app.add_middleware(ExceptionMiddlware, timeout=pyscore_settings.timeout)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(ResponseValidationError, validation_exception_handler)
app.add_exception_handler(ValidationError, validation_exception_handler)

# appに新規スコアと途上スコアのrouterを含める
app.include_router(new_pyscore.router)
app.include_router(repeater_pyscore.router)


# /staticディレクトリをappにマウントして認識させる
app.mount("/static", StaticFiles(), name="static")


# デフォルトでSwagger UIが起動しないため、Swagger UIの静的ファイルを生成
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/api/swagger-ui-bundle.js",
        swagger_css_url="/api/swagger-ui.css",
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()
