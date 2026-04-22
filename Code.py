

import pandas as pd
import numpy as np

df = pd.read_csv("/content/austin_housing_data.csv")

print("Shape:", df.shape)
display(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nSplit counts:")
print(df["split"].value_counts(dropna=False))

print("\nPrice summary by split:")
display(df.groupby("split")[["price", "target"]].agg(["count", "mean"]))

train_df = df[df["split"] == "train"].copy()
test_df  = df[df["split"] == "test"].copy()

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

y = train_df["price"].copy()

X_train_full = train_df.drop(columns=["price"])
X_test_full = test_df.drop(columns=["price"], errors="ignore")

display(X_train_full.head())

X_train = X_train_full.copy()
X_test = X_test_full.copy()

num_cols = X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

bad_num = {"id"}
bad_cat = {"split", "homeImage", "description"}

num_cols = [col for col in num_cols if col in X_train.columns and col not in bad_num]
cat_cols = [col for col in cat_cols if col in X_train.columns and col not in bad_cat]

print("Numeric columns:", num_cols)
print("\nCategorical columns:", cat_cols)

id_train = X_train["id"].copy()
id_test = X_test["id"].copy()


train_image_names = X_train["homeImage"].copy()
test_image_names = X_test["homeImage"].copy()

train_descriptions = X_train["description"].fillna("").copy()
test_descriptions = X_test["description"].fillna("").copy()

drop_cols = ["id", "split", "homeImage", "description"]
X_train = X_train.drop(columns=drop_cols, errors="ignore")
X_test = X_test.drop(columns=drop_cols, errors="ignore")


X_train["sqft_per_bed"] = X_train["livingAreaSqFt"] / (X_train["numOfBedrooms"] + 1)
X_test["sqft_per_bed"] = X_test["livingAreaSqFt"] / (X_test["numOfBedrooms"] + 1)

X_train["bath_bed_ratio"] = X_train["numOfBathrooms"] / (X_train["numOfBedrooms"] + 1)
X_test["bath_bed_ratio"] = X_test["numOfBathrooms"] / (X_test["numOfBedrooms"] + 1)

X_train["total_rooms"] = X_train["numOfBedrooms"] + X_train["numOfBathrooms"]
X_test["total_rooms"] = X_test["numOfBedrooms"] + X_test["numOfBathrooms"]

X_train["home_age"] = 2024 - X_train["yearBuilt"]
X_test["home_age"] = 2024 - X_test["yearBuilt"]

X_train["lot_to_living_ratio"] = X_train["lotSizeSqFt"] / (X_train["livingAreaSqFt"] + 1)
X_test["lot_to_living_ratio"] = X_test["lotSizeSqFt"] / (X_test["livingAreaSqFt"] + 1)

for col in num_cols:
    if col in X_train.columns:
        train_median = X_train[col].median()
        X_train[col] = X_train[col].fillna(train_median)
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(train_median)

for col in cat_cols:
    if col in X_train.columns:
        X_train[col] = X_train[col].fillna("Unknown")
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna("Unknown")

X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

print("Encoded train shape:", X_train.shape)
print("Encoded test shape:", X_test.shape)

y_log = np.log1p(y)

print("Any nulls in X_train?", X_train.isnull().sum().sum())
print("Any nulls in X_test?", X_test.isnull().sum().sum())
print("Target shape:", y_log.shape)

from google.colab import drive
drive.mount('/content/drive')

import os

IMAGE_DIR = "/content/drive/MyDrive/25-26/images"

print("Folder exists:", os.path.exists(IMAGE_DIR))

if os.path.exists(IMAGE_DIR):
    print("Sample files:", os.listdir(IMAGE_DIR)[:10])

print("Sample values from train_image_names:")
print(train_image_names.head(10).tolist())

!pip install -q transformers pillow torch torchvision

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

all_image_files = os.listdir(IMAGE_DIR)

image_lookup = {}
for file in all_image_files:
    full_path = os.path.join(IMAGE_DIR, file)
    base = os.path.basename(file).strip()
    stem = os.path.splitext(base)[0].strip()

    image_lookup[base.lower()] = full_path
    image_lookup[stem.lower()] = full_path

def get_full_image_path(image_name, image_dir=IMAGE_DIR):
    if pd.isna(image_name):
        return None

    name = str(image_name).strip()
    base = os.path.basename(name).strip()
    stem = os.path.splitext(base)[0].strip()

    if os.path.exists(name):
        return name

    if base.lower() in image_lookup:
        return image_lookup[base.lower()]
    if stem.lower() in image_lookup:
        return image_lookup[stem.lower()]

    for key, val in image_lookup.items():
        if stem.lower() == key or stem.lower() in key or key in stem.lower():
            return val

    return None

train_path_found = train_image_names.apply(lambda x: get_full_image_path(x) is not None)
test_path_found = test_image_names.apply(lambda x: get_full_image_path(x) is not None)

print("Train path-found rate:", train_path_found.mean())
print("Test path-found rate:", test_path_found.mean())
print("Train rows with found path:", train_path_found.sum(), "/", len(train_path_found))
print("Test rows with found path:", test_path_found.sum(), "/", len(test_path_found))

unique_images = pd.Series(
    pd.concat([train_image_names, test_image_names]).dropna().unique()
)

print("Using all unique images:", len(unique_images))

sample_check = pd.DataFrame({
    "homeImage": train_image_names.head(30).tolist()
})

sample_check["resolved_path"] = sample_check["homeImage"].apply(get_full_image_path)
sample_check["path_exists"] = sample_check["resolved_path"].apply(
    lambda x: os.path.exists(x) if x is not None else False
)

display(sample_check)

from tqdm.auto import tqdm
import torch.nn.functional as F

BATCH_SIZE = 32

embedding_rows = []

for i in tqdm(range(0, len(unique_images), BATCH_SIZE)):
    batch_names = unique_images[i:i+BATCH_SIZE]

    images = []
    valid_names = []

    for name in batch_names:
        path = get_full_image_path(name)
        if path is not None:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_names.append(name)
            except Exception:
                continue

    if len(images) == 0:
        continue

    inputs = clip_processor(images=images, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        vision_outputs = clip_model.vision_model(pixel_values=pixel_values)
        features = vision_outputs.pooler_output
        features = F.normalize(features, p=2, dim=-1)

    features = features.cpu().numpy()

    for name, emb in zip(valid_names, features):
        row = {"homeImage": name}
        for j, val in enumerate(emb):
            row[f"img_emb_{j}"] = float(val)
        embedding_rows.append(row)

image_embeddings_df = pd.DataFrame(embedding_rows)
print(image_embeddings_df.shape)
display(image_embeddings_df.head())

print("Number of unique images in train+test:", len(pd.concat([train_image_names, test_image_names]).dropna().unique()))
print("Number of rows in image_embeddings_df:", len(image_embeddings_df))
print("Number of unique homeImage values in image_embeddings_df:", image_embeddings_df["homeImage"].nunique())

image_embeddings_df.to_csv("/content/image_embeddings.csv", index=False)
print("Saved /content/image_embeddings.csv")

train_model_df = pd.DataFrame({"homeImage": train_image_names}).reset_index(drop=True)
test_model_df = pd.DataFrame({"homeImage": test_image_names}).reset_index(drop=True)

train_model_df = train_model_df.merge(image_embeddings_df, on="homeImage", how="left")
test_model_df = test_model_df.merge(image_embeddings_df, on="homeImage", how="left")

train_model_df = train_model_df.drop(columns=["homeImage"])
test_model_df = test_model_df.drop(columns=["homeImage"])

print(train_model_df.shape, test_model_df.shape)
display(train_model_df.head())

img_cols = [c for c in train_model_df.columns if c.startswith("img_emb_")]

train_embedded = train_model_df[img_cols].notna().any(axis=1)
test_embedded = test_model_df[img_cols].notna().any(axis=1)

print("Train rows with merged embeddings:", train_embedded.sum(), "/", len(train_embedded))
print("Test rows with merged embeddings:", test_embedded.sum(), "/", len(test_embedded))

from sklearn.decomposition import PCA

X_train_reset = X_train.reset_index(drop=True)
X_test_reset = X_test.reset_index(drop=True)

train_model_df = train_model_df.fillna(0)
test_model_df = test_model_df.fillna(0)

img_cols = [c for c in train_model_df.columns if c.startswith("img_emb_")]

nonzero_rows = (train_model_df[img_cols].abs().sum(axis=1) > 0)
print("Rows with image signal:", nonzero_rows.sum(), "/", len(nonzero_rows))
print("Match rate:", nonzero_rows.mean())

pca = PCA(n_components=15, random_state=42)

train_img_pca = pca.fit_transform(train_model_df[img_cols])
test_img_pca = pca.transform(test_model_df[img_cols])

train_img_pca_df = pd.DataFrame(
    train_img_pca,
    columns=[f"img_pca_{i}" for i in range(15)]
)

test_img_pca_df = pd.DataFrame(
    test_img_pca,
    columns=[f"img_pca_{i}" for i in range(15)]
)

print("Image PCA shapes:", train_img_pca_df.shape, test_img_pca_df.shape)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=3
)

train_text_tfidf = tfidf.fit_transform(train_descriptions)
test_text_tfidf = tfidf.transform(test_descriptions)

text_svd = TruncatedSVD(n_components=50, random_state=42)

train_text_svd = text_svd.fit_transform(train_text_tfidf)
test_text_svd = text_svd.transform(test_text_tfidf)

train_text_svd_df = pd.DataFrame(
    train_text_svd,
    columns=[f"text_svd_{i}" for i in range(50)]
)

test_text_svd_df = pd.DataFrame(
    test_text_svd,
    columns=[f"text_svd_{i}" for i in range(50)]
)

X_train_multi = pd.concat(
    [X_train_reset, train_img_pca_df, train_text_svd_df],
    axis=1
)

X_test_multi = pd.concat(
    [X_test_reset, test_img_pca_df, test_text_svd_df],
    axis=1
)

print("Multimodal train shape:", X_train_multi.shape)
print("Multimodal test shape:", X_test_multi.shape)

from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_pred = np.zeros(len(X_train_multi))
test_pred = np.zeros(len(X_test_multi))

fold_rmse_log = []
fold_rmse = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_multi), 1):
    print(f"\nStarting fold {fold}")

    X_tr = X_train_multi.iloc[train_idx]
    X_val = X_train_multi.iloc[val_idx]
    y_tr = y_log.iloc[train_idx]
    y_val = y_log.iloc[val_idx]

    model = XGBRegressor(
    n_estimators=4000,
    learning_rate=0.015,
    max_depth=4,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.6, 
    reg_alpha=0.3,
    reg_lambda=2.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    val_pred_log = model.predict(X_val)
    oof_pred[val_idx] = val_pred_log

    test_pred += model.predict(X_test_multi) / 5

    rmse_log = np.sqrt(mean_squared_error(y_val, val_pred_log))
    rmse_orig = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred_log)))

    fold_rmse_log.append(rmse_log)
    fold_rmse.append(rmse_orig)

    print(f"Fold {fold} RMSE (log): {rmse_log:.6f}")
    print(f"Fold {fold} RMSE (original): {rmse_orig:.4f}")

overall_rmse_log = np.sqrt(mean_squared_error(y_log, oof_pred))
overall_rmse = np.sqrt(mean_squared_error(np.expm1(y_log), np.expm1(oof_pred)))

print("\nCV mean RMSE (log):", np.mean(fold_rmse_log))
print("CV mean RMSE (original):", np.mean(fold_rmse))
print("OOF RMSE (log):", overall_rmse_log)
print("OOF RMSE (original):", overall_rmse)

y_test_pred_log = test_pred
y_test_pred = np.expm1(y_test_pred_log)

print(y_test_pred[:5])
print("Number of predictions:", len(y_test_pred))

submission = pd.read_csv("/content/HW__Austin_Housing_Price_Prediction_template.csv")

print(submission.shape)
display(submission.head())

importance_df = pd.DataFrame({
    "feature": X_train_multi.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

display(importance_df.head(20))

pred_df = pd.DataFrame({
    "id": id_test.values,
    "target": y_test_pred
})

submission = submission[["id"]].merge(pred_df, on="id", how="left")

display(submission.head())

print("Missing targets:", submission["target"].isna().sum())

print("Template rows:", len(submission))
print("Prediction rows:", len(y_test_pred))

submission.to_csv("submission.csv", index=False)

submission.to_csv("/content/drive/MyDrive/submission.csv", index=False)
print("Saved to Drive")

from google.colab import files
files.download("/content/drive/MyDrive/submission.csv")
