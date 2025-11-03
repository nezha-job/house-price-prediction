import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle

# Ma'lumotlarni yuklash
df = pd.read_csv("train.csv", encoding='latin1')
df.fillna(0, inplace=True)

# Foydalaniladigan ustunlar
features = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
X = df[features]
y = df['SalePrice']

# Trening va test to‘plami
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kuchli model: XGBoost
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Natijani tekshiramiz
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE (o‘rtacha xatolik): ${mae:,.2f}")
print("✅ model.pkl fayli yaratildi.")

# Saqlash
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
