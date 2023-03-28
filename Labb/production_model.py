import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# filepaths.
test_samples_path = "Labb/model/test_samples.csv"
model_path = "Labb/model/model.pkl"

# load model and samples.
test_samples = pd.read_csv(test_samples_path)
model = joblib.load(model_path)

# spliting samples in to test data. 
X_test, y_test = test_samples.drop(columns="cardio", axis= 1), test_samples["cardio"]
# predict on test samples 
y_pred = model.predict(X_test)
# gets prob for prediction.
y_prob = model.predict_proba(X_test)

# creates a dataframe with prediction and probability class.
d = {"prediction": y_pred,"probability class 0": y_prob[:, 0], "probability class 1": y_prob[:, 1]}
df_predictions = pd.DataFrame(d)

# saves predictions as a csv file.
df_predictions.to_csv("Labb/model/prediction.csv", index= False)