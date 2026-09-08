import pickle
from pathlib import Path

from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.utils import save_model


def main(model_name: str):
    root_dir = Path().cwd()
    models_dir = root_dir / "models"

    # モデルの読み込み
    with open(models_dir / f"{model_name}.pkl", "rb") as f:
        model = pickle.load(f)

    booster = model.get_booster()  # booster呼び出し
    original_feature_names = booster.feature_names  # スコアアイテム名の呼び出し
    original_feature_types = booster.feature_types

    # スコアアイテム名をf0~fnに変更（n=スコアアイテム数）
    if original_feature_names is not None:
        onnx_converter_conform_feature_names = [f"f{i}" for i in range(len(original_feature_names))]
        booster.feature_names = onnx_converter_conform_feature_names

    # スコアアイテムの型が'i'(bool)の場合、'int'に変更
    if original_feature_types is not None:
        onnx_converter_conform_feature_types = ["int" if type == "i" else type for type in original_feature_types]
        booster.feature_types = onnx_converter_conform_feature_types

    # モデルをONNX形式に変換 -> .onnxで保存
    initial_type = [("float_input", FloatTensorType([None, 22]))]
    onx = convert_xgboost(model, initial_types=initial_type)
    save_model(onx, models_dir / f"{model_name}.onnx")


if __name__ == "__main__":
    main("repeater_model_v2")
