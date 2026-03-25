import numerapi

napi = numerapi.NumerAPI()
VERSION = "v5.2"

print("Downloading the V5.2 dataset...")

# These standard files are now natively int8 compressed!
napi.download_dataset(f"{VERSION}/train.parquet", "train.parquet")
napi.download_dataset(f"{VERSION}/validation.parquet", "validation.parquet")
napi.download_dataset(f"{VERSION}/features.json", "features.json")

print("Download complete.")
