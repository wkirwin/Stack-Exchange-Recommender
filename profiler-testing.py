from stackrecommender import Recommender
from stacksite import StackSite
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
import warnings
try:
    import cPickle as pkl
except:
    import pickle as pkl
import sys, time
import bz2
from tqdm import *

def split_dfs(df_dict, n_folds = 5):
	"""
	Splits questions_df, comments_df into test/train (test/total = 1/n_folds) 
	and slices answers_df, users_df accordingly. Tags_df is not split.

	Returns: split_df = {'test':test_df_dict, 'validate':validate_df}
	"""

	df_names = ['questions', 'comments']
	df_lengths = {name:len(df_dict[name]) for name in df_names}

	# generate the train/test split index arrays
	# I'll just use the first set of indices for my single-fold stats
	folds = {name:KFold(df_lengths[name], n_folds=n_folds) for name in df_names}

	# generate the train/test dataframes
	train_dfs = {name:[] for name in df_names}
	test_dfs = {name:[] for name in df_names}
	for name in ['questions', 'comments']:
	    for train, test in folds[name]:
	        train_dfs[name] = df_dict[name].ix[train]
	        test_dfs[name] = df_dict[name].ix[test]

	# split answers_df according to the train/test split of questions_d
	answers_df = df_dict['answers'].copy()
	train_qids = set(train_dfs['questions'].index.unique()) # indices are already unique
	test_qids = set(test_dfs['questions'].index.unique()) # but df.index is mutable and not hashable
	train_dfs['answers'] = answers_df[answers_df.parent_id.isin(train_qids)]
	test_dfs['answers'] = answers_df[answers_df.parent_id.isin(test_qids)]

	# pack the training sets into dictionaries of dataframes for the recommender
	df_names.append('answers')
	train_df_dict = {}
	train_df_dict = {name:train_dfs[name] for name in df_names}
	train_df_dict['tags'] = df_dict['answers']
	train_df_dict['users'] = df_dict['users']

	# users and scores are lists of users/scores lists (one list for each fold)
	# question_ids is a list of lists of the corresponding question_ids
	# questions is a list of lists of questions (one list for each fold)

	users = test_dfs['answers'].user_id
	scores = test_dfs['answers'].score
	question_ids = test_dfs['answers'].parent_id
	questions = [test_dfs['questions'][['title','question','tags']].ix[qid] for qid in question_ids]
	    
	# make a list of dataframes
	validate_df = pd.DataFrame(data=questions, index=question_ids)
	validate_df['user_id'] = pd.Series(data=users.values, index=question_ids)
	validate_df['score'] = pd.Series(data=scores.values, index=question_ids)
	validate_df['predicted_score'] = 0.
	validate_df['support'] = 0.

	return {'train':train_df_dict, 'validate':validate_df}


if __name__ == '__main__':

	site_name = 'datascience.stackexchange.com'

	print "\n == One-fold Validation of %s ==\n" % site_name
	site = StackSite(site_name)
	site.load()

	print "%s loaded." % site_name
	sys.stdout.flush()

	n_folds = 4
	tv_dfs = split_dfs(site.df_dict(), n_folds = n_folds)
	train_df_dict = tv_dfs['train']
	validate_df = tv_dfs['validate']

	print "Dataframes split (%d-fold)." % n_folds
	sys.stdout.flush()

	print "Training..."
	sys.stdout.flush()

	recommender = Recommender(site_name,train_df_dict)
	recommender.train(iterations = 1000, passes = 1)

	print "Predicting..."
	sys.stdout.flush()

	for j in trange(len(validate_df), leave=True):
	    pred_score, supp = recommender.predicted_score(validate_df.user_id.irow(j), 
	                                             validate_df[['title','question','tags']].irow(j))
	    validate_df['predicted_score'][j] = pred_score
	    validate_df['support'][j] = supp

	print "\nWriting results..."
	with bz2.BZ2File('data/'+site_name+'/one-fold-results.bzpkl','wb') as f:
		pkl.dump(validate_df,f)
	    
	print "\nDone!\n"