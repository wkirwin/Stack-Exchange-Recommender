from stacksite import StackSite
from stackrecommender import Recommender
import sys, time

def make_recommendations(site_name, N=10, method='LF'):

	print "Recommedations for %s" % site_name
	sys.stdout.flush()
	
	start_time = time.time()
	print " ...loading site"
	sys.stdout.flush()
	site = StackSite(site_name)
	site.load()

	print " ...building recommender"
	sys.stdout.flush()
	rc = Recommender(site_name, df_dict = site.df_dict())
	rc.train()
	rc.save('full')
	rc.load('full')

	print " ...making recommendations"
	rc.recommend_all()

	end_time = time.time()
	print "Done in %0.2fs!" % (end_time - start_time)

	return


if __name__ == '__main__':

	with open("site_names.csv", "r") as f:
		site_names = f.read().splitlines()

	if (len(sys.argv) != 2) or (sys.argv[1] not in site_names):
		print "syntax: python makeRecommender.py <site_name>"
		print "<site_name> should be a valid StackExchange site from site_names.csv."

	else:
		make_recommendations(sys.argv[1])
