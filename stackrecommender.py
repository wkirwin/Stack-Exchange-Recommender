import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import gensim
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import operator
import warnings
import sys
import distance
try:
    import cPickle as pkl
except:
    import pickle as pkl
import gzip

class Recommender(object):
    """
    Initialization:
    site_name = "<site>.stackexchange.com" : name of the site
    df_dict = StackSite(site_name).df_dict()
    W2Vmodel = binary word2vec model (should probably preload GoogleNews-vectors-negative300.bin and pass it)
    """
    
    def __init__(self, site_name, df_dict = None, W2Vmodel = None):
        # site_name : all files will be read from/saved to ./data/site_name
        # loading the binary word2vec model is *slow*... it is much better to load it once
        #   and pass it to different recommenders
        self.site_name = site_name

        # initialize the dataframes
        if df_dict is not None:
            self.users_df = df_dict['users']
            self.questions_df = df_dict['questions']
            self.answers_df = df_dict['answers']
            self.tags_df = df_dict['tags']

        # I can't recommend to deleted accounts
        self.users_df = self.users_df.drop('-999')

        # numerical indexing
        self.question_idx = pd.Series(data=range(len(self.questions_df)), index=self.questions_df.index)
        self.tags_idx = pd.Series(data=range(len(self.tags_df)), index=self.tags_df.name)

        # initialize the gensim objects
        self.W2Vmodel = W2Vmodel

        # question-vector arrays
        self.questionW2V = None # an ndarray of size (n_questions, n_topics) to hold the LDA vectors of the questions
        self.tags = None # an ndarray of size (n_questions, n_tags) to hold the boolean tag vectors for the questions

        # similarity matrices
        self.question_similarity = None
        self.tag_similarity = None
        #self.tfidf_similarity = None

        # latent factor vectors and indices
        try:
            self.lf_question = pkl.load(gzip.GzipFile('data/'+self.site_name+'/pfeatures.gzpkl', 'rb'))
            self.lf_user = pkl.load(gzip.GzipFile('data/'+self.site_name+'/ufeatures.gzpkl', 'rb'))
            self.lf_uidx = pkl.load(open('data/'+self.site_name+'/uidx.pkl', 'rb'))
            self.lf_qidx = pkl.load(open('data/'+self.site_name+'/midx.pkl', 'rb'))
        except:
            print "Error loading latent factors. Are trained vectors available?"
            self.lf_user = None
            self.lf_question = None
            self.lf_uidx = None
            self.lf_qidx = None

        return


    # (re)train the recommender
    def train(self):
        """
        Computes word2vec and tag vectors for each question, as well as the resulting cosine-similarity matrices.
        """

        stops = set(stopwords.words("english"))

        # an ndarray of size (n_questions, n_topics) to hold the word2vec-vectors of the questions
        self.questionW2V = np.array(map(lambda x: self.q2v(x, stops),
                                    (self.questions_df.title + " " + self.questions_df.question).values))

        # an ndarray of size (n_questions, n_tags) to hold the boolean tag vectors for the questions
        self.tags = self.post2tagvec(self.questions_df.tags) 

        self.question_similarity = cosine_similarity(self.questionW2V, self.questionW2V)
        # maybe use Jaccard similarity?
        self.tag_similarity = np.dot(self.tags, self.tags.T)

        return
    
    # save a trained model    
    def save(self, name):
        """
        Save a trained model to disk.
        """

        print "Saving the model..."
        
        try:
            pkl.dump(self.questionW2V, gzip.GzipFile('data/'+self.site_name+'/'+name+'_questionW2V', 'wb'))
            pkl.dump(self.tags, gzip.GzipFile('data/'+self.site_name+'/'+name+'_tagVec', 'wb'))

            pkl.dump(self.question_similarity, gzip.GzipFile('data/'+self.site_name+'/'+name+'_questionSim', 'wb'))
            pkl.dump(self.tag_similarity, gzip.GzipFile('data/'+self.site_name+'/'+name+'_tagSim', 'wb'))

        except:
            print "Error saving."

        return
    
    
    # load a trained model
    def load(self, name):
        """
        Loads a trained model from disk.
        """

        print "Loading the model..."
        sys.stdout.flush()
        
        try:
            self.question_similarity = pkl.load(gzip.GzipFile('data/'+self.site_name+'/'+name+'_questionSim', 'rb'))
            self.tag_similarity = pkl.load(gzip.GzipFile('data/'+self.site_name+'/'+name+'_tagSim', 'rb'))
            self.questionW2V = pkl.load(gzip.GzipFile('data/'+self.site_name+'/'+name+'_questionW2V', 'rb'))
            self.tags = pkl.load(gzip.GzipFile('data/'+self.site_name+'/'+name+'_tagVec', 'rb'))

            print "Successfully loaded the recommender."
            sys.stdout.flush()

        except:
            print "Error loading the model."
            
        return
    
    
    # helper function to convert post text (question, answer, comment, title)
    #   to a list of words

    def q2BoW(self, qtext, stops = None):
        """
        Converts qtext to a list of words, excluding stopwords and restricting to words in W2Vmodel.vocab.
        """

        if not stops:
            stops = set(stopwords.words("english"))

        qtext = BeautifulSoup(qtext).getText()
        alpha_only = re.sub("[^a-zA-Z]", " ", qtext)
        words = alpha_only.split()
        meaningful_words = [w for w in words if ((w not in stops) and (w in self.W2Vmodel.vocab))]

        return meaningful_words
    
    def q2v(self, qtext, stops=None):
        """
        Converts a question to the normalized mean word2vec of the meaninfgul words in
        qtext.
        """
        
        v = [self.W2Vmodel[word] for word in self.q2BoW(qtext, stops)]
        return gensim.matutils.unitvec(np.array(v).mean(axis=0))
    
    def post2tagvec(self, tags):
        """
        Converts the list of tags or the Series of lists of tags to (an ndarray of) boolean tag vectors.
        """

        def fill_vec_from_taglist(taglist):
            bool_vec = np.zeros(len(self.tags_df))
            for tag in taglist:
                try:
                    bool_vec[self.tags_idx.ix[tag]] = 1
                except:
                    pass
            return bool_vec

        if isinstance(tags, list): # got a single list
            return fill_vec_from_taglist(tags)
        else:
            bool_vecs = np.zeros((len(tags), len(self.tags_df)))
            for j in xrange(len(tags)):
                try:
                    bool_vecs[j] = fill_vec_from_taglist(tags[j])
                except:
                    # there are apparently some tags which are not officiall (i.e. in tags_df)
                    # we ignore them
                    pass
            return bool_vecs

    # aggregated similarity between questions (title+question+tags)
    def post_sim(self, q1, q2):
        """Use for computing similarities of new posts. Not yet implemented."""
        return

    
    # k-NN for the questions based on weighted cosine + boolean tag similarity
    def k_nearest_questions(self, question_id, qdf, k = 7, similarity = 'W2V'):
        """
        Returns the k nearest neighbors of the question among those questions in questions_df.

        Inputs
        ======
        question : an index of self.question_df
        qdf : a subindex of self.questions_df.index
        k : (optional) number of neighbors
        similarity : (str, optional) similarity matrix to use. Currently supported: 
                                        'W2V' : cosine similarity of LDA vectors
                                        'tag' : (dot product of boolean tag vectors)
        """

        if similarity == 'tag':
            sim_matrix = self.tag_similarity
        elif similarity == 'W2V':
            sim_matrix = self.question_similarity

        if len(qdf) == 0: 
            return None, None
        else: 
            similarities =  pd.Series(sim_matrix[self.question_idx[question_id],qdf],
                                      index=qdf)

        similarities.sort(ascending=False)
        k = min(k, len(qdf))
        similarities = similarities.head(k+1)

        # if the question is in the supplied list, don't return the question itself as a nearest neighbor
        if question in similarities.index:
            similarities = similarities.drop(question)

        return similarities.head(k)

    
    # k-NN stars predictor
    def predict_stars(self,user_id, question_id, k = 7, force_predict = False, method = 'LF'):
        """
        Predicts the stars of the user's response to the question.
        
        Inputs
        ======
        user_id
        question_id = an index of self.questions_df or a 3-tuple (title, question, tags)
        k : (optional, default = 7) based on k-NN
        force_predict : (optional, default = False) if true, returns the model prediction even
                            when a user has answered a question. Otherwise, returns the actual score.
        method : (str, optional, default = 'LF') model to use. Supported values are:
                 'LF' : use precomputed latent factors
                 'LF+W2V' : LF model + LDA-similarity-based collaborative filter (CF) for the residues (not yet implemented)
                 'LF+tag' : LF model + tag-similarity based CF for the residues (NYI)
                 'LF+Tfidf' : LF model + tfidf-3gram-similarity based CF for the residues (NYI)

        Returns
        =======
        (float) the user_id's predicted stars for answering the given question

        If the user has answered the question, returns the user's answer stars."""

        if method == "CF": # not possible to use this yet... I should implement it soon though
            user_answered_questions_idx = self.question_idx[self.answers_df[self.answers_df.user_id == user_id].parent_id.unique()]

            if len(user_answered_questions_idx) == 0:
                #print "This user hasn't answered any questions yet!"
                return None, None

            if not force_predict:
                if question_id in user_answered_questions_idx:
                    Yuq = self.answers_df[self.answers_df.parent_id == self.questions_df.ix[question_id]].stars[0]
                    #print "User already answered this question with an answer stars of %d." % Yuq
                    return Yuq, 0

            user_answered_questions_df = self.questions_df.ix[user_answered_questions_idx]

            # print user_answered_questions_df
            # sys.stdout.flush()

            user_base_stars = self.answers_df[self.answers_df.user_id == user_id].stars.mean()
            
            if np.isnan(user_base_stars): # this probably should never happen...if it did, we would've already returned above
                # user hasn't answered any question yet: set base stars to the mean stars of all answers
                # by users who have answered a single question
                user_base_stars = self.avg_new_user_stars

            Yuq = 0.
            total_sim = 0.

            top_qs = self.k_nearest_questions(question_id, user_answered_questions_idx, 
                                      k = k)

            for j in range(len(top_qs)):
                q = top_qs.index[j]
                sim = top_qs.values[j]
                q_avg_other_stars = self.answers_df[(self.answers_df.parent_id == self.questions_df.index[q]) 
                                                    & (self.answers_df.user_id != user_id)].stars.mean()

                if np.isnan(q_avg_other_stars):
                    #print "Got nan."
                    q_avg_other_stars = self.avg_new_user_stars
                
                Yuq += sim * (self.answers_df[(self.answers_df.parent_id == self.questions_df.index[q]) 
                                                & (self.answers_df.user_id == user_id)].stars.ix[0] - q_avg_other_stars)
                total_sim += sim
                
            if total_sim == 0:
                # user has not answered *any* similar questions
                return user_base_stars, 0

            n_supp = np.sum([x != 0 for x in top_qs.values])

            #if np.isnan(user_base_stars + Yuq/total_sim):
                #print "We have nan problem: ", user_base_stars, Yuq, total_sim

            return (user_base_stars + Yuq/total_sim), n_supp
        
        elif method == 'LF':

            if not (set(question_id) <= set(self.lf_qidx.index)):
                #print "You appear to be asking about a new question. This isn't implemented yet."
                return None

            if user_id not in self.lf_uidx.index:
                return None

            uidx = self.lf_uidx[user_id]
            qidx = self.lf_qidx[question_id]

            normed_stars = pd.Series(np.dot(self.lf_user[uidx], self.lf_question[qidx].T),
                                     index = question_id)

            prediction = normed_stars + self.users_df.mean_stars.ix[user_id] \
                                      + self.questions_df.mean_stars.ix[question_id] \
                                      - self.answers_df.stars.mean()

            return prediction
    
    # the recommender
    
    def recommend_questions(self, user_id, N = 7, k = 7, method = 'LF'):
        """
        Inputs
        ======
        user_id
        N : (optional) number of recommendations to make
        k : k-NN parameter
        method : (option, default 'LF') how to predict scores, as follows:
                 'LF' : use precomputed latent factors
                 'LF+W2V' : LF model + LDA-similarity-based collaborative filter (CF) for the residues (not yet implemented)
                 'LF+tag' : LF model + tag-similarity based CF for the residues (NYI)
                 'LF+Tfidf' : LF model + tfidf-3gram-similarity based CF for the residues (NYI)


        Returns
        =======
        dataframe : N rows from questions_df describing the top N recommended
        user_id-unanswered questions.
        """

        #if self.LDAmodel == None:
        #    print "Train the recommender first!"
        #    return None
        
        if user_id not in self.answers_df.user_id.tolist():
            # print "User has not answered any questions... default to global recommendations."
            return None



        answered_q_ids = self.answers_df[self.answers_df.user_id == user_id].parent_id.unique()
        unanswered_qdf = self.questions_df.drop(answered_q_ids)[['title','question','tags']]
        predictable_qdf = unanswered_qdf.ix[self.lf_qidx.index]
        if len(unanswered_qdf) == 0:
            # print "There are no unanswered questions to recommend for user %s!!!" % user_id
            return None

        decorated_stars = pd.DataFrame(self.predict_stars(user_id, predictable_qdf.index),
                                    columns = ['predicted']).dropna().sort('predicted',ascending=False)

        return decorated_stars.head(N).index

    # make recommendations for all users and store the list of recommended question_id's in self.user_df
    def recommend_all(self, N = 10, method = 'LF', save = True, pgdb_url = None):
        """
        Makes recommendations for all users.
        save = True : save the user and question dataframes to gzipped pickled files
        savePSQL = 'PostgreSQL Database URL' : save user and question dataframes to tables (site)_udf, (site)_qdf
                    in the given postgres database.
        """
        
        def rq(uid):
            if uid == '-999':
                return None
            else:
                return self.recommend_questions(uid, N = N, method = method)
        
        self.users_df['recommendations'] = self.users_df.index.map(rq)

        if save:
            with gzip.GzipFile('data/'+self.site_name+'/userdf.gzpkl', 'wb') as f:
                pkl.dump(self.users_df, f)

            with gzip.GzipFile('data/'+self.site_name+'/qdf.gzpkl', 'wb') as f:
                pkl.dump(self.questions_df, f)

        if pgdb_url is not None:
            from sqlalchemy import create_engine

            # convert Index arrays and None to lists
            def safelist(x):
                if x is None:
                    return []
                else:
                    return list(x)
            
            # convert NaN stars to -1
            def safestars(x):
                if np.isnan(x):
                    return -1.
                else:
                    return x

            # clean up the user DataFrame for db insertion
            udf = self.users_df.copy()
            udf.recommendations = udf.recommendations.apply(safelist)
            udf.mean_stars = udf.mean_stars.apply(safestars)
            udf = udf.reset_index()

            # clean up the question DataFram for db insertion
            qdf = self.questions_df.copy()
            qdf.mean_stars = qdf.mean_stars.apply(safestars)
            qdf = qdf.reset_index()

            engine = create_engine(pgdb_url)
            prefix = self.site_name.split('.')[0]

            udf.to_sql(prefix+'_udf', engine, if_exists='replace')
            qdf.to_sql(prefix+'_qdf', engine, if_exists='replace')
            
        return   

    # convert a precomputed list of recommended question_id's to a dataframe of questions
    def convert_recommendations(self, user_id, N = 10):
        if self.users_df.ix[user_id].recommendations is None or len(self.users_df.recommendations.ix[user_id]) == 0:
            return None
        else:
            return self.questions_df[['title','question','tags']].ix[self.users_df.recommendations.ix[user_id]].head(N)

    # a wrapper to print the list of recommendations
    def print_recommendations(self, user_id, N = 7, k = 7, method = 'LF', IPython = False):
        """
        A wrapper for recommend_quesions: prettier output.
        Inputs
        ======
        user_id
        N : (optional) number of recommendations to make
        k : k-NN parameter
        method : (str, optional, default 'LF')
        Prints the list of recommended questions.
        Returns: None
        """

        if IPython:
            from IPython.display import display, HTML

        recommendations = self.recommend_questions(user_id, N = N, k = k, method = method)

        for qid in recommendations:
            print "\nQuestion %d" % (list(recommendations).index(qid)+1)
            print "=========="
            if IPython:
                display(HTML(self.questions_df.title.ix[qid]))
                print
                display(HTML(self.questions_df.question.ix[qid]))
            else:
                print "# "+self.questions_df.title.ix[qid]+" #"
                print
                print self.questions_df.question.ix[qid]
            print "-"*83

        return None

    def similar_questions(self, title, question, N=10):
        """
        Returns the top N most similar questions (using W2V vectors).
        """
        
        qvec = self.q2v(title+" "+question)
        
        similarities = pd.Series(cosine_similarity(qvec,self.questionW2V)[0], index=self.question_idx.index)
        similarities.sort(ascending=False)
        
        return similarities.head(N).index