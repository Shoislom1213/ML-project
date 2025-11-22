import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import joblib

# 🔹 Transformer class
class FullTransformer:
    def __init__(self, drop_row=5347):
        self.drop_row = drop_row
        self.scaler = StandardScaler()
    
    def fit_transform(self, X, y=None):
        X = X.copy()
        # 1️⃣ Numeric conversion & filtering
        X['price'] = pd.to_numeric(X['price'], errors='coerce')
        X['size'] = pd.to_numeric(X['size'], errors='coerce').fillna(0).astype(int)
        X = X[(X['size'] < 400) & (X['rooms'] < 9) & (X['price'] < 400000)]
        
        # 2️⃣ Feature engineering
        X['price_per_kv'] = X['price'] / X['size']
        X['size_per_room'] = X['size'] / X['rooms']
        X["floor_ratio"] = X["level"] / X["max_levels"]
        X['mean_price_per_district'] = X.groupby('district')['price'].transform('mean')
        
        # 3️⃣ Numeric + one-hot encoding
        df_num = X.select_dtypes(include=[np.number])
        df_one_hot = pd.get_dummies(X[['district']], drop_first=True)
        df_ready = pd.concat([df_num, df_one_hot], axis=1)
        
        # 4️⃣ Drop specific row
        if self.drop_row in df_ready.index:
            df_ready = df_ready.drop(self.drop_row)
        
        # 5️⃣ Scaling
        X_scaled = self.scaler.fit_transform(df_ready.drop('price', axis=1))
        y = df_ready['price'].values
        return X_scaled, y

# 🔹 Data load
data = pd.read_csv('https://raw.githubusercontent.com/anvarnarz/praktikum_datasets/main/housing_data_08-02-2021.csv')
transformer = FullTransformer()
x, y = transformer.fit_transform(data)

# 🔹 CatBoost model (tuning natijalar bilan)
CB_model = CatBoostRegressor(
    verbose=0,
    random_state=42,
    learning_rate=0.03,   # tuning natijasidan
    depth=4,              # tuning natijasidan
    iterations=1500,       # tuning natijasidan
    l2_leaf_reg=3          # tuning natijasidan
)

# 🔹 Modelni fit qilish
CB_model.fit(x, y)

# 🔹 Saqlash
joblib.dump(CB_model, "catboost_model.pkl")

# 🔹 Modelni qayta yuklash va bashorat qilish
loaded_model = joblib.load("models/catboost_model.pkl")
y_pred = loaded_model.predict(x)

# 🔹 Natijalarni ko‘rish
print("Birinchi 10 bashorat:", y_pred[:10])
