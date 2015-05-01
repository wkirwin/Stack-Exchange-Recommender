import numpy as np
import pandas as pd
import re, string
from nltk.corpus import stopwords
from pattern import web
from bs4 import BeautifulSoup
import os, sys
import gzip
try:
    import cPickle as pkl
except:
    import pickle as pkl

class StackSite(object):
    
    def __init__(self, name, download=False):
        self.site_name = name
        self.questions_df = None
        self.answers_df = None
        self.tags_df = None
        self.comments_df = None
        self.users_df = None
        self.fulldf = None
        
        if download:
            dir = 'data/'+self.site_name
            if not os.path.exists(dir):
                os.makedirs(dir)
            files = os.listdir(dir)
            for file in files:
                if file.endswith(".xml") or file.endswith(".7z"):
                    os.remove(os.path.join(dir,file))
            self.download_site()


        expected_files = ["Posts.xml", "Users.xml", "Tags.xml", "Comments.xml"]
        try:
            for filename in expected_files:
                f = open('data/'+name+'/'+filename,"r")
                f.close()
        except:
            print "%s is missing: %s" % ('data/'+name+'/',filename)
            download = raw_input("Do you want to download the files from the Internet Archive? [y/n] ")
            if download == 'y':
                self.download_site()
            else:
                print "You're going to have a bad time."
        
        return

    def download_site(self):
        """
        Downloads .7z file from the Internet Archive and decompresses it to 'data/self.site_name/'.
        It is assumed to contain (at least): Posts.xml, Users.xml, Tags.xml, Comments.xml.
        """

        import urllib
        from pyunpack import Archive

        # reporthook taken from: http://stackoverflow.com/questions/13881092/download-progressbar-for-python-3
        def reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                percent = readsofar * 1e2 / totalsize
                s = "\r%5.1f%% %*d / %d" % (
                    percent, len(str(totalsize)), readsofar, totalsize)
                sys.stderr.write(s)
                if readsofar >= totalsize: # near the end
                    sys.stderr.write("\n")
            else: # total size is unknown
                sys.stderr.write("read %d\n" % (readsofar,))

        directory = 'data/'+self.site_name+'/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        try:
            urllib.urlretrieve ("https://archive.org/download/stackexchange/"+self.site_name+".7z", 
                                directory+self.site_name+".7z",
                                reporthook)
            sys.stderr.flush()
            print "Download complete."
            sys.stdout.flush()

        except:
            print "Failed to download the requested site."

        print "Extracting files and cleaning up."
        sys.stdout.flush()


        Archive(directory+self.site_name+".7z").extractall('data/'+self.site_name)
        os.remove(directory+self.site_name+".7z")

        sys.stdout.flush()

        return

    def save(self):
        """
        Save the dataframes to pickled gzip files in data/self.site_name.
        """

        dfs = {'questions':self.questions_df,
               'answers':self.answers_df,
               'tags':self.tags_df,
               'users':self.users_df,
               'comments':self.comments_df}

        if sum([(df is None) for df in dfs.values()]) > 0:
            # some dataframes are missing
            generate_now = raw_input("Some dataframes are missing. Generate them now? [y/n] ")
            if generate_now == 'y':
                dfs = self.generate_dfs()
            else:
                print "OK, but there isn't much to do with missing dataframes."
                return

        with gzip.GzipFile('data/'+self.site_name+'/dataframes.gzpkl', 'wb') as f:
            pkl.dump(dfs, f)

        with gzip.GzipFile('data/'+self.site_name+'/fulldf.gzpkl', 'wb') as f:
            pkl.dump(self.fulldf, f)

        print "Dataframes successfully saved."
        sys.stdout.flush()

        return

    def load(self):
        """
        Loads previously saved dataframes.
        """

        if not os.path.exists('data/'+self.site_name+'/dataframes.gzpkl'):
            generate_now = raw_input("I can't find any saved dataframes to load. Generate them now? [y/n] ")
            if generate_now == 'y':
                dfs = self.generate_dfs()
                return
            else:
                print "OK, but there isn't much to do with missing dataframes."
                return
        with gzip.GzipFile('data/'+self.site_name+'/dataframes.gzpkl','rb') as f:
            dfs = pkl.load(f)

            self.answers_df = dfs['answers']
            self.questions_df = dfs['questions']
            self.users_df = dfs['users']
            self.tags_df = dfs['tags']
            self.comments_df = dfs['comments']

        with gzip.GzipFile('data/'+self.site_name+'/fulldf.gzpkl','rb') as f:
            self.fulldf = pkl.load(f)

        print "Dataframes successfully loaded."
        sys.stdout.flush()

        return

    def convert_questions(self):
        """
        Convert Posts.xml to the required questions_df.
        """
        
        posts_dom = web.Element(file('data/'+self.site_name+"/Posts.xml").read())
        
        # Warning: super-hacky code follows!
        qids, uids, tags, titles, bodies, creation_dates = [],[],[],[],[],[]

        for row in posts_dom.by_tag('row'):

            if row.attributes['posttypeid'] == '1': # get the questions
                qids.append(row.attributes['id'])
                if 'owneruserid' in row.attributes.keys():
                    uids.append(row.attributes['owneruserid'])
                else: uids.append(u'-999')
                if 'tags' in row.attributes.keys():
                    tags.append(re.sub('[^-a-zA-Z]',' ',row.attributes['tags']).strip().split())
                else: tags.append([u''])
                titles.append(row.attributes['title'].encode('unicode-escape').replace(r'\n',''))
                bodies.append(row.attributes['body'].encode('unicode-escape').replace(r'\n',''))
                creation_dates.append(pd.to_datetime(row.attributes['creationdate']))
                
        self.questions_df = pd.DataFrame(data={'post_id':qids,
                              'user_id':uids,
                              'title':titles,
                              'tags':tags,
                              'question':bodies,
                              'date':creation_dates},
                        columns=['post_id','user_id','title','tags','question','date'])
            
        self.questions_df = self.questions_df.set_index('post_id') # index by (unique) post_id
        
        return
    
    def convert_answers(self):
        """
        Convert Posts.xml to the required answers_df.
        """
        
        posts_dom = web.Element(file('data/'+self.site_name+"/Posts.xml").read())
        
        aids, uids, parentids, bodies, scores = [],[],[],[], []

        for row in posts_dom.by_tag('row'):

            if row.attributes['posttypeid'] == '2': # get the answers
                aids.append(row.attributes['id'])
                if 'owneruserid' in row.attributes.keys():
                    uids.append(row.attributes['owneruserid'])
                else: uids.append(u'-999')
                scores.append(int(row.attributes['score']))
                parentids.append(row.attributes['parentid'])
                bodies.append(row.attributes['body'].encode('unicode-escape'))

        self.answers_df = pd.DataFrame(data={'post_id':aids,
                                  'user_id':uids,
                                  'parent_id':parentids,
                                  'score':scores,
                                  'answer':bodies},
                            columns=['post_id','user_id','parent_id','score','answer'])
        self.answers_df = self.answers_df.set_index('post_id') # index by (unique) post_id

        return

    def convert_users(self):
        """
        Convert Users.xml to the required users_df.
        """
        
        users_dom = web.Element(file('data/'+self.site_name+'/Users.xml').read())
        
        user_ids, displaynames, reputations = ['-999'], ['(account deleted)'], [0]
        
        for row in users_dom.by_tag('row'):
            user_ids.append(row.attributes['id'])
            displaynames.append(row.attributes['displayname'].encode('unicode-escape'))
            reputations.append(int(row.attributes['reputation']))

        self.users_df = pd.DataFrame(data = {'user_id':user_ids, 
                                'displayname':displaynames,
                                'reputation':reputations},
                        columns = ['user_id','displayname','reputation'])
        self.users_df = self.users_df.set_index('user_id')
        
        return
        
        
    def convert_comments(self):
        """
        Convert Comments.xml to the required comments_df.
        """
        
        comments_dom = web.Element(file('data/'+self.site_name+'/Comments.xml').read())
        
        cids, pids, uids, bodies = [], [], [], []
        for row in comments_dom.by_tag('row'):
            if 'userid' in row.attributes.keys():
                cids.append(row.attributes['id'])
                pids.append(row.attributes['postid'])
                uids.append(row.attributes['userid'])
                bodies.append(row.attributes['text'].encode('unicode-escape'))

        self.comments_df = pd.DataFrame(data={'comment_id':cids,
                                 'post_id':pids,
                                 'user_id':uids,
                                 'comment':bodies},
                           columns=['comment_id','post_id','user_id','comment'])
        self.comments_df = self.comments_df.set_index('comment_id')
        
        return
    
    def convert_tags(self):
        """
        Convert Comments.xml to the required comments_df.
        """
        
        tags_dom = web.Element(file('data/'+self.site_name+"/Tags.xml").read())

        ids, names, counts = [], [], []
        for row in tags_dom.by_tag('row'):
            ids.append(row.attributes['id'])
            names.append(row.attributes['tagname'].encode('unicode-escape'))
            counts.append(int(row.attributes['count']))
            
        self.tags_df = pd.DataFrame(data={'tag_id':ids, 'name':names, 'count':counts},
                               columns = ['tag_id', 'name', 'count'])
        self.tags_df = self.tags_df.set_index('tag_id')
        
        return

    def generate_fulldf(self):
        """
        Makes fulldf (required by the Spark LF Recommender).
        """

        users_df = self.users_df.drop('-999')
        answers_df = self.answers_df[self.answers_df.user_id != '-999']
        questions_df = self.questions_df.copy()

        def star(score):
            cutoffs = np.array([-np.inf, 0, 2, 7, 17])
            stars = len(np.where(cutoffs < score)[0]) - 1
            return stars

        self.answers_df['stars'] = self.answers_df.score.apply(star)

        # make the full database of (user,question) pairs
        fulldf = pd.DataFrame(data = {'user_id': answers_df.user_id.values,
                                       'question_id': answers_df.parent_id.values,
                                       'question': (questions_df.title.ix[answers_df.parent_id] + " " + questions_df.question.ix[answers_df.parent_id]).values,
                                       'tags': questions_df.tags.ix[answers_df.parent_id].values,
                                       'score': answers_df.score.values,
                                       'id_pair': zip(answers_df.user_id.values, answers_df.parent_id.values)})
        fulldf['stars'] = fulldf.score.apply(star)

        umeans = fulldf.groupby('user_id').stars.mean()
        mmeans = fulldf.groupby('question_id').stars.mean()

        # store the user and item means
        self.users_df['mean_stars'] = None
        self.users_df['mean_stars'] = umeans

        self.questions_df['mean_stars'] = None
        self.questions_df['mean_stars'] = mmeans

        fulldf['user_mean'] = fulldf.user_id.apply(lambda uidx: umeans.ix[uidx])
        fulldf['item_mean'] = fulldf.question_id.apply(lambda qidx: mmeans.ix[qidx])

        self.fulldf = fulldf

        return


    
    def generate_dfs(self):
        """
        Converts the .xml files of the SE site into a dict of dataframes for use in the recommender.
        """

        self.convert_questions()
        self.convert_answers()
        self.convert_users()
        self.convert_comments()
        self.convert_tags()
        self.generate_fulldf()
        
        return

    def df(self, name):
        """
        Returns self.(name)_df.

        Inputs: name must be one of the following: 'questions', 'answers', 'tags', 'users', 'comments'
        """

        dfs = {'questions':self.questions_df,
               'answers':self.answers_df,
               'tags':self.tags_df,
               'users':self.users_df,
               'comments':self.comments_df}
        try:
            return dfs[name]
        except:
            print "%s is not one of the site's dataframes."
            return None

    def df_dict(self):
        """
        Returns the dictionary of dataframes necessary to initialize the recommender.
        """

        dfs = {'questions':self.questions_df,
               'answers':self.answers_df,
               'tags':self.tags_df,
               'users':self.users_df,
               'comments':self.comments_df,
               'fulldf':self.fulldf}

        if sum([(df is None) for df in dfs.values()]) > 0:
            # need to initialize the DataFrames
            self.generate_dfs()

        dfs = {'questions':self.questions_df,
               'answers':self.answers_df,
               'tags':self.tags_df,
               'users':self.users_df,
               'comments':self.comments_df,
               'fulldf':self.fulldf}

        return dfs