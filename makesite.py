import sys
from stacksite import StackSite
import warnings

warnings.simplefilter('ignore', 'UserWarning')



if __name__=="__main__":

	with open("site_names.csv", "r") as f:
		site_names = f.read().splitlines()

	error_string = "syntax: python makesite.py <site_name>\n where <site_name> is a valid StackExchange site from site_names.csv."
	if len(sys.argv) != 2:
		print error_string
	
	elif sys.argv[1] not in site_names:
		print error_string
		
	else:
		site = StackSite(sys.argv[1], download=True)
		site.generate_dfs()
		site.save()

		print "Successfully downloaded and build dataframes for %s" % sys.argv[1]