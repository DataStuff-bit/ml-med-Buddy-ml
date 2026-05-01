import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import dump
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    recall_score,
    f1_score
)

def train_model():
    try:
        load_dotenv()

        PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()
        DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_DIR") / os.getenv("DATASET_NAME")
        MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR")/ os.getenv("MODEL_NAME")
        LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR")/ os.getenv("LOG_NAME")

        TARGET_COL = os.getenv("TARGET_COL")
        TEST_SIZE = float(os.getenv("TEST_SIZE"))
        RANDOM_STATE = int(os.getenv("RANDOM_STATE"))

        MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True,exist_ok=True)

        logging.basicConfig(
            level = logging.INFO,
            format = "%(asctime)s - %(levelname)s - %(message)s",
            handlers = [
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH)
            ]
        )

        # Load data
        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Dataset loaded with shape: {df.shape}")

        # Seperate x and y
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        # Create a signature for each feature row to prevent duplicate leakage to test set
        row_signatures = pd.util.hash_pandas_object(X,index=False)

        # Group based split (Same logic as notebook)
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        train_idx,test_idx = next(gss.split(X,y,groups=row_signatures))

        X_train,X_test = X.iloc[train_idx],X.iloc[test_idx]
        y_train,y_test = y.iloc[train_idx],y.iloc[test_idx]

        logging.info(f"Data split into train and test sets with train size : {X_train.shape} & test size : {X_test.shape} and test size: {TEST_SIZE}")
        
        # Best Params from Notebook
        best_rf = RandomForestClassifier(
            random_state = RANDOM_STATE,
            n_jobs = -1,
            bootstrap = True,
            ccp_alpha = 0.0017,
            max_depth = 5,
            max_features = 'sqrt',
            max_samples = 0.6,
            min_samples_leaf = 11,
            min_samples_split = 30,
            n_estimators = 1119
        )
    
        #Keep scaler in pipeline to match notebook structure
        pipeline = Pipeline(
            steps = [
                ("scaler",StandardScaler()),
                ("model",best_rf)
            ]
        )

        pipeline.fit(X_train,y_train)
        logging.info("Model training completed")

        # Evaluation
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_acc = accuracy_score(y_train,y_train_pred)
        test_acc = accuracy_score(y_test,y_test_pred)

        train_recall = recall_score(y_train,y_train_pred)
        test_recall = recall_score(y_test,y_test_pred)

        train_f1 = f1_score(y_train,y_train_pred)
        test_f1 = f1_score(y_test,y_test_pred)

        logging.info(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
        logging.info(f"Train Recall: {train_recall:.4f}, Test Recall: {test_recall:.4f}")
        logging.info(f"Train F1 Score: {train_f1:.4f}, Test F1 Score: {test_f1:.4f}")
        logging.info("Classification Report on Train Set:")
        logging.info("\n" + classification_report(y_train,y_train_pred))
        logging.info("Classification Report on Test Set:")
        logging.info("\n" + classification_report(y_test,y_test_pred))
        dump(pipeline,MODEL_PATH)
        logging.info(f"Model saved at {MODEL_PATH}")

        logging.info("Training process completed successfully")

    except Exception as e:
        print(f"Training failed:{e}")
        logging.exception(f"Training failed:{e}")
        raise

if __name__ == "__main__":
    train_model()