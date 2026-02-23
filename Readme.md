# classification_roboflow

Roboflowの画像分類データセットを使って、`MobileNetV2`をローカル環境でファインチューニングする構成です。

## 1. セットアップ

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. 環境変数を設定

```powershell
Copy-Item .env.example .env
```

`.env` を編集:

- `ROBOFLOW_API_KEY`
- `ROBOFLOW_WORKSPACE`
- `ROBOFLOW_PROJECT`
- `ROBOFLOW_VERSION`

## 3. 学習実行（ローカル）

```powershell
.\run_train.ps1
```

`run_train.ps1` は `.env` を読み込み、`train_mobilenet.py` を起動します。

## 4. 既にローカルにデータセットがある場合

`.env` に `DATASET_DIR` を設定すると、Roboflowから再ダウンロードせずに学習します。

```env
DATASET_DIR=C:\path\to\dataset
```

## 主なオプション（直接実行時）

```powershell
python train_mobilenet.py --workspace <workspace> --project <project> --version <version> --epochs-head 5 --epochs-finetune 10
```

- `--img-size` 入力解像度（デフォルト `224`）
- `--batch-size` バッチサイズ（デフォルト `32`）
- `--finetune-layers` 解凍するMobileNet上位レイヤ数（デフォルト `30`）
- `--dataset-dir` ローカルデータセットを使うパス
- `--output-dir` 出力先（デフォルト `artifacts`）

## 出力

- `artifacts/best.keras`: 検証精度ベストの重み
- `artifacts/mobilenetv2_finetuned.keras`: 最終モデル
- `artifacts/labels.txt`: クラス名一覧
