# prediction.py
import inspect
from typing import Dict, List

import numpy as np
from onnxruntime import InferenceSession, SessionOptions

from ..settings import pyscore_settings


class Prediction:
    """スコアリングモデルで予測し、閾値に応じてランク分けする"""

    def __init__(self, model_path, feature_name_list: List[str], ths_list: List[float]) -> None:
        # ONNX Runtimeのオプション設定
        options = SessionOptions()
        options.intra_op_num_threads = pyscore_settings.intra_op_num_threads  # スレッド数
        # ONNXモデルのSession起動
        self.sess = InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
        # スコアアイテム名の定義
        self.feature_name_list = feature_name_list
        # 閾値の定義
        self.ths_list = ths_list

    def main(self, preprocessed_record: Dict) -> Dict:
        """スコアリングモデルによる予測とランク分け

        Parameters
        ----------
        preprocessed_record: Dict
            前処理済みレコード

        Returns
        ----------
        Dict
            {'id': リクエストに紐づくID, 'pred_score': 予測未収率, 'rank': ランク}
        """
        try:
            id = preprocessed_record["id"]
            X = self.make_features(preprocessed_record)
            y_prob = self.predict(X)
            rank = self.output_rank(y_prob)
        except KeyError as e:
            error_loc = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
            error_msg = f"{e.__class__.__name__}: {str(e)} is missing."
            return error_loc, error_msg
        except Exception as e:
            error_loc = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
            error_msg = f"{e.__class__.__name__}: {str(e)}"
            return error_loc, error_msg
        else:
            return {"id": id, "score": y_prob, "rank": rank}

    def make_features(self, preprocessed_record: Dict) -> np.array:
        """前処理済みレコードからスコアアイテムのみを抽出

        Parameters
        ----------
        preprocessed_record: Dict
            前処理済みの申込レコード

        Returns
        ----------
        np.array
            スコアアイテムの値が格納されている配列
        """
        try:
            # 前処理済みレコードからスコアアイテムのみを抽出
            feature_list = [preprocessed_record[feature_name] for feature_name in self.feature_name_list]
            # モデルのinputに合わせてfloat32に変換
            X_arr = np.array(feature_list, dtype=np.float32)
        except KeyError as e:
            # いずれかのスコアアイテムが含まれていない場合
            error_loc = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
            error_msg = f"{e.__class__.__name__}: {str(e)} is missing."
            return error_loc, error_msg
        except Exception as e:
            error_loc = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
            error_msg = f"{e.__class__.__name__}: {str(e)}"
            return error_loc, error_msg
        else:
            # 正常時はスコアアイテムの2次元配列を返す
            return X_arr.reshape(1, -1)

    def predict(self, X: np.array) -> float:
        """スコアリングモデルを用いて予測未収率を出力

        Parameters
        ----------
        X: np.array
            スコアアイテム
        sess: rt.InferenceSession
            ONNX Runtimeのセッション

        Returns
        ----------
        float
            予測未収率
        """
        try:
            # ONNXモデルのinput名をSessionから呼び出し
            input_name = self.sess.get_inputs()[0].name
            # 予測
            _, probs = self.sess.run(None, {input_name: X})
        except Exception as e:
            error_loc = f"{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
            error_msg = {str(e)}
            return error_loc, error_msg
        else:
            # 正常時は予測未収率を返す
            return float(probs[0, 1])

    def output_rank(self, y_prob: float) -> int:
        """閾値に応じて予測未収率をランク分け

        Parameters
        ----------
        y_prob: float
            予測未収率

        Returns
        ----------
        int
            ランク
        """

        rank = np.digitize(y_prob, self.ths_list)

        return int(rank)
