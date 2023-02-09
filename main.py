import traceback
import numpy as np
import pandas as pd
import gresearch_crypto
# from lightgbm import LGBMRegressor
import xgboost as xgb

TRAIN_CSV = '/kaggle/input/g-research-crypto-forecasting/train.csv'
ASSET_DETAILS_CSV = '/kaggle/input/g-research-crypto-forecasting/asset_details.csv'

#Load Data
df_train = pd.read_csv(TRAIN_CSV)
df_train.head()
df_asset_details = pd.read_csv(ASSET_DETAILS_CSV).sort_values("Asset_ID")
df_asset_details

#Extract Features and Train Model
# Two new features from the competition tutorial
def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])

def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']

# A utility function to build features from the original df
# It works for rows to, so we can reutilize it.
def get_features(df):
    df_feat = df[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].copy()
    df_feat['Upper_Shadow'] = upper_shadow(df_feat)
    df_feat['Lower_Shadow'] = lower_shadow(df_feat)
    return df_feat

def get_Xy_and_model_for_asset(df_train, asset_id):
    df = df_train[df_train["Asset_ID"] == asset_id]
    
    # TODO: Try different features here!
    df_proc = get_features(df)
    df_proc['y'] = df['Target']
    df_proc = df_proc.dropna(how="any")
    
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]
    
    model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=13,
        learning_rate=0.004,
        subsample=0.9,
        colsample_bytree=0.7,
        missing=-999,
        random_state=2020,
        tree_method='gpu_hist'  # THE MAGICAL PARAMETER
    )
    model.fit(X, y)

    return X, y, model
  
Xs = {}
ys = {}
models = {}

for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
    print(f"Training model for {asset_name:<16} (ID={asset_id:<2})")
    try:
        X, y, model = get_Xy_and_model_for_asset(df_train, asset_id)    
        Xs[asset_id], ys[asset_id], models[asset_id] = X, y, model
    except: 
        traceback.print_exc()
        Xs[asset_id], ys[asset_id], models[asset_id] = None, None, None
        
#Predict
env = gresearch_crypto.make_env()
iter_test = env.iter_test()

for i, (df_test, df_pred) in enumerate(iter_test):
    for j , row in df_test.iterrows():
        
        if models[row['Asset_ID']] is not None:
            try:
                model = models[row['Asset_ID']]
                x_test = get_features(row)
                y_pred = model.predict(pd.DataFrame([x_test]))[0]
                df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = y_pred
            except:
                df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = 0
                traceback.print_exc()
        else: 
            df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = 0
        
    env.predict(df_pred)
