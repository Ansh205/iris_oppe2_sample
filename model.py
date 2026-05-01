import mlflow

# 🔥 Set MLflow server (VERY IMPORTANT)
mlflow.set_tracking_uri("http://34.170.61.111:5000")

def load_model():
    """
    Loads the best model from MLflow Model Registry
    (Production stage)
    """
    #model = mlflow.pyfunc.load_model("models:/iris_model/Production")
    model=mlflow.pyfunc.load_model("models:/iris_model@production")
    return model