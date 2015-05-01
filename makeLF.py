import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, Rating
from scipy.sparse import coo_matrix
import multiprocessing as mp
import cPickle as pkl
import gzip
import os, sys
import warnings
warnings.filterwarnings('ignore')


# Configure the environment
os.environ['SPARK_HOME'] = '/usr/local/spark'
# sys.path.insert(0, '/usr/local/spark/python')

# Create a variable for our root path
SPARK_HOME = os.environ['SPARK_HOME']

# Add the PySpark/py4j to the Python Path
sys.path.insert(0, os.path.join(SPARK_HOME, "python", "build"))
sys.path.insert(0, os.path.join(SPARK_HOME, "python"))

if __name__ == "__main__":

	if len(sys.argv) != 2:
		print "Usage: python makeLF.py <site_name>"
		print " where <site_name> is a valide StackExchange site from site_names.csv."

	else:
		site_name = sys.argv[1]

		# load the site data (from the prebuilt dataframe)
		print "  ...loading the site"
		sys.stdout.flush()
		with gzip.GzipFile("data/"+site_name+"/fulldf.gzpkl", "r") as f:
			fulldf = pkl.load(f)

		# instantiate a local SparkContext
		print "  ...initiating the SparkContext"
		sys.stdout.flush()
		n_cores = mp.cpu_count()
		sc = SparkContext('local['+str(n_cores)+']', 'pyspark')

		# parallelize the normalized data
		print "  ...parallelizing the data"
		sys.stdout.flush()
		mu = fulldf.stars.mean()
		data = sc.parallelize(fulldf[['user_id','question_id','stars','user_mean','item_mean']].as_matrix())
		normed_ratings = data.map(lambda row: Rating(int(row[0]), int(row[1]), row[2] - row[3] - row[4] + mu))

		# build the LF model
		print "  ...building the model"
		sys.stdout.flush()
		rank = 40
		numIterations = 30
		model = ALS.train(normed_ratings, rank, numIterations)

		# extract the user and product features into numpy arrays
		print "  ...extracting the factors"
		sys.stdout.flush()
		uf = sorted(model.userFeatures().collect())
		pf = sorted(model.productFeatures().collect())

		U = np.vstack([a[1] for a in uf])
		V = np.hstack([np.array(a[1]).reshape(rank,1) for a in pf]).T

		# record the user/product_id <-> index correspondences
		uids = [str(a[0]) for a in uf]
		mids = [str(a[0]) for a in pf]
		user_idx = pd.Series(range(len(uids)), index=uids)
		item_idx = pd.Series(range(len(mids)), index=mids)

		# make ALL the predictions
		# print "  ...making ALL the predictions"
		# sys.stdout.flush()
		# bigdata = sc.parallelize([(int(uid), int(mid)) for uid in uids for mid in mids])

		# predictions = zip(*model.predictAll(bigdata).map(lambda r: (user_idx.ix[str(r[0])], item_idx.ix[str(r[1])], r[2])).collect())
		# P = coo_matrix((predictions[2], (predictions[0], predictions[1]))).todense()

		# stop the SparkContext
		sc.stop()

		# write the arrays to disk
		print "  ...writing to disk"
		sys.stdout.flush()
		with gzip.GzipFile("data/"+site_name+"/ufeatures.gzpkl", "wb") as f:
		    pkl.dump(U, f)
		with gzip.GzipFile("data/"+site_name+"/pfeatures.gzpkl", "wb") as f:
		    pkl.dump(V, f)
		# with gzip.GzipFile("data/"+site_name+"/predictions.gzpkl", "wb") as f:
		#	pkl.dump(P, f)
		with open("data/"+site_name+"/uidx.pkl", "wb") as f:
		    pkl.dump(user_idx, f)
		with open("data/"+site_name+"/midx.pkl", "wb") as f:
		    pkl.dump(item_idx, f)

		# add the site to the list of ready sites
		ready_sites = set([]);
		if not os.path.exists('static'):
			os.makedirs('static')
		open('static/ready_sites.csv', 'a').close()
		with open('static/ready_sites.csv', 'r') as f:
			ready_sites = set(f.read().splitlines())
		ready_sites.add(site_name)
		with open('static/ready_sites.csv', 'w') as f:
			[f.write(site+"\n") for site in ready_sites]

		print "Successfully created latent-factor vectors for %s." % site_name