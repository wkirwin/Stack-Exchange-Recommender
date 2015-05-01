from stacksite import StackSite
from stackrecommender import Recommender
import sys, time

def make_recommendations(site_name, pgdb_url, N=10, method='LF'):
	
	start_time = time.time()
	print " ...loading site"
	sys.stdout.flush()
	site = StackSite(site_name)
	site.load()

	print " ...loading recommender"
	sys.stdout.flush()
	rc = Recommender(site_name, df_dict = site.df_dict())
	#rc.train()
	#rc.save('full')
	#rc.load('full')

	print " ...making recommendations"
	rc.recommend_all(pgdb_url=pgdb_url)

	end_time = time.time()
	print "Done in %0.2fs!" % (end_time - start_time)

	return


if __name__ == '__main__':

	with open("site_names.csv", "r") as f:
		site_names = f.read().splitlines()

	print "Making recommender for %s with Postgres hook %s" % (sys.argv[1], sys.argv[2])

	if (len(sys.argv) != 3):
		print "syntax: python makeRecommender.py <site_name> <PostgresDB URL>"
	elif  (sys.argv[1] not in site_names):
		print "<site_name> should be a valid StackExchange site from site_names.csv."
	else:
		make_recommendations(sys.argv[1], sys.argv[2])
