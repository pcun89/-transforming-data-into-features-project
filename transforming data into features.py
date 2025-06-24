import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import numpy as np

# Step 1: Create Sample Dataset
data = {
    'height_cm': [150, 160, 170, 180],
    'weight_kg': [55, 65, 75, 85],
    'gender': ['female', 'male', 'female', 'male'],
    'signup_date': pd.to_datetime(['2020-01-01', '2021-06-15', '2021-07-20', '2022-03-10'])
}

df = pd.DataFrame(data)

# Step 2: Normalization and Standardization
normalizer = MinMaxScaler()
df[['height_norm', 'weight_norm']] = normalizer.fit_transform(
    df[['height_cm', 'weight_kg']])

scaler = StandardScaler()
df[['height_std', 'weight_std']] = scaler.fit_transform(
    df[['height_cm', 'weight_kg']])

# Step 3: Handle Categorical Data (One-Hot Encoding)
# drop first to avoid multicollinearity
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_gender = encoder.fit_transform(df[['gender']])
gender_cols = encoder.get_feature_names_out(['gender'])

df_encoded = pd.DataFrame(encoded_gender, columns=gender_cols, index=df.index)
df = pd.concat([df, df_encoded], axis=1)

# Step 4: Datetime Feature Engineering
df['signup_month'] = df['signup_date'].dt.month
df['signup_day'] = df['signup_date'].dt.day
df['signup_year'] = df['signup_date'].dt.year
df['signup_weekday'] = df['signup_date'].dt.weekday

# Step 5: Dimensionality Reduction with PCA
pca = PCA(n_components=1)
df['pca_feature'] = pca.fit_transform(df[['height_std', 'weight_std']])

# Final Output1
print(df)
