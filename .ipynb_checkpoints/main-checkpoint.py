def main():
    print("Hello from numerai-init-project!")

    import pandas as pd
    import json
    import numerapi
    import os
    from dotenv import load_dotenv
    # 1. Load the hidden keys from the .env file
    load_dotenv()
    
    napi = numerapi.NumerAPI()
    VERSION = "v5.2"
    
    public_id = os.getenv("NUMERAI_PUBLIC_ID")
    secret_key = os.getenv("NUMERAI_SECRET_KEY")
    
    # 2. Initialize the API connection
    print("Authenticating with Numerai...")
    napi = numerapi.NumerAPI(public_id, secret_key)
    
    # 3. Verify the connection by checking your profile
    profile = napi.get_account()
    print(f"Successfully connected! Logged in as: {profile['username']}")
    
    # (Your data downloading, model prediction, and upload code follows here...)
    
    
    # 2. Read the JSON metadata to get the feature names
    with open(f"features.json", "r") as f:
        feature_metadata = json.load(f)
    
    # 3. Extract the "small" feature set (Fixing the official typo here)
    small_features = feature_metadata["feature_sets"]["small"]
    
    # We need the 'era' (for time-series validation), our features, and the 'target'
    columns_to_read = ["era"] + small_features + ["target"]
    
    print(f"Loading {len(columns_to_read)} columns out of 2000+...")
    
    # 4. Load ONLY the required columns into pandas
    training_data = pd.read_parquet(f"train.parquet", columns=columns_to_read)
    
    # Verify the memory footprint
    mem_usage = training_data.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Data loaded successfully! Memory usage: {mem_usage:.2f} MB")


    from sklearn.decomposition import PCA
    import numpy as np
    
    # Extract just the feature columns
    feature_cols = [c for c in training_data.columns if c.startswith("feature_")]
    X_train = training_data[feature_cols]
    
    # Fit Global PCA (extracting the most dominant component)
    global_pca = PCA(n_components=1)
    global_pca.fit(X_train)
    
    # This is the primary global market vector
    global_vector = global_pca.components_[0]

    era_vectors = {}

    # Iterate through unique eras
    for era in training_data['era'].unique():
        # Isolate data for this specific era
        X_era = training_data[training_data['era'] == era][feature_cols]
        
        # Fit PCA just for this era
        era_pca = PCA(n_components=1)
        era_pca.fit(X_era)
        
        # Store the primary vector for this era
        era_vectors[era] = era_pca.components_[0]

    similarities = {}

    for era, era_vector in era_vectors.items():
        # Calculate the dot product
        dot_product = np.dot(global_vector, era_vector)
        
        # We take the absolute value because the sign (+/-) of an eigenvector is arbitrary in PCA
        similarities[era] = np.abs(dot_product)
    
    # Example: Check the alignment of the first era
    for era in list(similarities.keys()):
        print(f"Similarity for {era}: {similarities[era]:.4f}")

    from scipy.stats import beta

    # Extract just the similarity scores into a numpy array
    sim_data = np.array(list(similarities.values()))
    
    # Fit the Beta distribution to our data
    # We force loc=0 and scale=1 because our absolute cosine similarities are strictly bounded between 0 and 1
    a, b, loc, scale = beta.fit(sim_data, floc=0, fscale=1)
    
    # Find the threshold where only 5% of the distribution falls to the left
    # PPF gives us the value at a specific percentile
    threshold = beta.ppf(0.05, a, b)
    
    print(f"Fitted Beta parameters: alpha={a:.2f}, beta={b:.2f}")
    print(f"Regime Change Threshold (Bottom 5%): {threshold:.4f}")
    
    # Identify the specific eras that are statistically anomalous
    anomalous_eras = [era for era, score in similarities.items() if score < threshold]
    print(f"Identified {len(anomalous_eras)} anomalous eras.")

    # 1. Isolate the data for the 34 anomalous eras
    anomalous_data = training_data[training_data['era'].isin(anomalous_eras)][feature_cols]
    
    # 2. Fit the "Crisis PCA" ONLY on the anomalous data
    crisis_pca = PCA(n_components=5) # Extracting the top 5 crisis factors
    crisis_pca.fit(anomalous_data)
    
    # 3. Apply (transform) the entire dataset using these crisis components
    # We use .transform(), NOT .fit_transform() here!
    crisis_features = crisis_pca.transform(training_data[feature_cols])
    
    # 4. Add these new features back to your main DataFrame
    for i in range(5):
        training_data[f'crisis_pca_{i}'] = crisis_features[:, i]


    import numpy as np
    import lightgbm as lgb
    from scipy.stats import spearmanr
    import optuna
    
    # Define the features we are training on (42 small features + 5 crisis components)
    all_training_features = small_features + [f'crisis_pca_{i}' for i in range(5)]


    import joblib
    #Only if training is not changed
    final_model = joblib.load('lgbm_sklearn_model.pkl')

    # 1. Initialize API and find the current round
    # (Assuming napi = numerapi.NumerAPI("YOUR_PUBLIC_KEY", "YOUR_SECRET_KEY") is already set)
    current_round = napi.get_current_round()
    print(f"Downloading live data for Round {current_round}...")
    
    # 2. Download the live dataset
    # The live data is much smaller than training data, so it downloads quickly
    napi.download_dataset("v5.2/live.parquet", "live.parquet")


    # 3. Load the live data
    # CRITICAL: We MUST include the 'id' column this time!
    live_columns = small_features
    live_data = pd.read_parquet("live.parquet", columns=live_columns).reset_index()
    
    print(f"Loaded {len(live_data)} live assets to predict.")
    
    # 4. Apply the exact same Feature Engineering (Crisis PCA)
    print("Applying Crisis PCA to live features...")
    live_crisis_features = crisis_pca.transform(live_data[small_features])
    
    for i in range(5):
        live_data[f'crisis_pca_{i}'] = live_crisis_features[:, i]
    
    # 5. Generate Predictions using your locked-in final model
    print("Generating predictions...")
    X_live = live_data[all_training_features]
    live_data['prediction'] = final_model.predict(X_live)
    
    # 6. Format the submission file
    # Numerai expects a CSV with exactly two columns: 'id' and 'prediction'
    submission_df = live_data[['id','prediction']].copy()
    
    # Ensure predictions are strictly between 0 and 1 (a requirement for Numerai)
    # Tree models usually output values within the target range, but it's a good safety clip
    submission_df['prediction'] = submission_df['prediction'].clip(0.0001, 0.9999)

    submission_filename = f"submission_round_{current_round}.csv"
    submission_df.to_csv(submission_filename, index=False)
    print(f"Saved predictions to {submission_filename}")

    # 7. Upload to the Tournament!
    print("Uploading to Numerai...")
    model_id = napi.get_models()['first_model41'] # Replace with your exact model name from the website
    napi.upload_predictions(submission_filename, model_id=model_id)
    
    print("Submission complete! 🎉")


if __name__ == "__main__":
    main()
