from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd, numpy as np
import mlflow, mlflow.sklearn
from sklearn.metrics import accuracy_score


data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# add synthetic gender (not for training)
X["gender"] = np.random.randint(0,2,len(X))

X_train, X_test, y_train, y_test = train_test_split(X.drop("gender",axis=1), y, test_size=0.2)


# -------------------------------
# 3. MLflow setup
# -------------------------------
mlflow.set_tracking_uri("http://34.170.61.111:5000")  # your MLflow server
mlflow.set_experiment("iris_oppe2")

# -------------------------------
# 4. Train model
# -------------------------------
with mlflow.start_run():

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # log metric
    mlflow.log_metric("accuracy", acc)

    # log model + register
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name="iris_model"
    )

    print("Training complete. Accuracy:", acc)