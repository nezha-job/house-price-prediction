import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Sun'iy ma'lumotlar (100 ta namunadan iborat)
np.random.seed(42)
overall_qual = np.random.randint(1, 11, 100)            # 1-10 ball
gr_liv_area = np.random.randint(500, 2500, 100)          # m²
garage_area = np.random.randint(0, 800, 100)             # m²
total_bsmt_sf = np.random.randint(0, 1500, 100)          # m²
full_bath = np.random.randint(1, 4, 100)                 # soni
year_built = np.random.randint(1950, 2022, 100)          # yil

# Narx formulasi
price = (
    overall_qual * 10000 +
    gr_liv_area * 50 +
    garage_area * 30 +
    total_bsmt_sf * 25 +
    full_bath * 5000 +
    (year_built - 1900) * 300 +
    np.random.randint(-20000, 20000, 100)  # shovqin
)

# DataFrame
df = pd.DataFrame({
    'OverallQual': overall_qual,
    'GrLivArea': gr_liv_area,
    'GarageArea': garage_area,
    'TotalBsmtSF': total_bsmt_sf,
    'FullBath': full_bath,
    'YearBuilt': year_built,
    'price': price
})

# Model
X = df.drop('price', axis=1)
y = df['price']

model = LinearRegression()
model.fit(X, y)

# Modelni saqlash
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ model.pkl fayli yaratildi — Flask formaga mos!")
