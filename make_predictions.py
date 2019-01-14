# USAGE
# python make_predictions.py --db ./data/features_test.hdf5 
#	--model ./model/dogs_vs_cats.pickle --output "./submissions/submission.csv"

# import the necessary packages
from sklearn.linear_model import LogisticRegression
import argparse
import pickle
import h5py
import pandas as pd

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True,
	help="path HDF5 database")
ap.add_argument("-m", "--model", required=True,
	help="path to trained model")
ap.add_argument("-o", "--output", required=True,
	help="path to export submission file")
args = vars(ap.parse_args())

# open the HDF5 database for reading the id of the files
db = h5py.File(args["db"], "r")
id = db["labels"]

print("[INFO] Loading Model")
model = pickle.load(open(args["model"], 'rb'))

# make predictionsl
print("[INFO] making predictions...")
preds = model.predict(db["features"])

submission = pd.DataFrame()
submission['id'] = id
submission['label'] = preds

submission = submission.sort_values(by='id')

submission.to_csv(args["output"],index=False)

# close the database
db.close()