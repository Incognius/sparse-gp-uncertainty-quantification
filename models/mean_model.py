import lightgbm as lgb
import joblib
import os

class EnergyMeanModel:
    def __init__(self):
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': 42,
            'learning_rate': 0.05,
            'num_leaves': 31
        }
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        self.model = lgb.train(
            self.params,
            dtrain,
            valid_sets=[dtrain, dval],
            valid_names=['train', 'valid'],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained or loaded yet.")
        return self.model.predict(X)

    def save(self, path='models/lgbm_mean_model.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, path='models/lgbm_mean_model.pkl'):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")