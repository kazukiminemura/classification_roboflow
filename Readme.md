# classification_roboflow

Roboflowの画像分類データセットを使って、`MobileNetV2`をファインチューニングする最小構成です。

## セットアップ

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 学習実行

`ROBOFLOW_API_KEY` を環境変数に入れるか、`--api-key`を指定してください。

```powershell
$env:ROBOFLOW_API_KEY="YOUR_API_KEY"
python train_mobilenet.py --workspace <workspace_slug> --project <project_slug> --version <dataset_version>
```

主なオプション:

- `--img-size` 入力解像度（デフォルト `224`）
- `--epochs-head` ヘッド学習エポック数（デフォルト `5`）
- `--epochs-finetune` 微調整エポック数（デフォルト `10`）
- `--finetune-layers` 解凍するMobileNet上位レイヤ数（デフォルト `30`）
- `--output-dir` 出力先（デフォルト `artifacts`）

## 出力

- `artifacts/best.keras`: 検証精度ベストの重み
- `artifacts/mobilenetv2_finetuned.keras`: 最終モデル
- `artifacts/labels.txt`: クラス名一覧
