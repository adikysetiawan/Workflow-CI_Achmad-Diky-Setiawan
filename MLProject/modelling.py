import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# MLflow Autolog (Sesuai instruksi Kriteria 3)
mlflow.autolog()

df = pd.read_csv('housing_processed.csv')
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    print("Model berhasil dilatih di dalam CI/CD!")