import numpy as np
import pandas as pd
import re, string
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import gensim
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import operator
import warnings
import sys
try:
    import cPickle as pkl
except:
    import pickle as pkl
import gzip

class Recommender(object):
    
    def __init__(self, site_name, df_dict = None):
        # site_name : all files will be read from/saved to ./data/site_name
        self.site_name = site_name

        # initialize the dataframes
        if df_dict is not None:
            self.questions_df = df_dict['questions']
            self.answers_df = df_dict['answers']
            self.tags_df = df_dict['tags']
            self.users_df = df_dict['users']
            self.comments_df = df_dict['comments']
            self.fulldf = df_dict['fulldf']

            # numerical indexing
            self.question_idx = pd.Series(data=range(len(self.questions_df)), index=self.questions_df.index)
            self.tags_idx = pd.Series(data=range(len(self.tags_df)), index=self.tags_df.name)

            # convert scores to stars and compute useful statistics
            def star(score):
                cutoffs = np.array([-np.inf, 0, 2, 7, 17])
                stars = int(len(np.where(cutoffs < score)[0]) - 1)
                return stars

            self.answers_df['stars'] = self.answers_df.score.apply(star)

            #    the average stars of first time answerers
            grp_answers_df = self.answers_df.groupby('user_id')

            # switched to median
            self.avg_new_user_stars = grp_answers_df.stars.mean()[grp_answers_df.stars.count() == 1].median()
            if np.isnan(self.avg_new_user_stars):
                warnings.warn("Warning: no users have answered only one question. This is likely a synthetic database.\n \
                    parameter avg_new_user_stars defaulting to 2.0")
                self.avg_new_user_stars = 2.0
        
        # initialize the gensim objects
        self.dictionary = None
        self.vectorized_corpus = None
        self.LDAmodel = None
        self.num_topics = 0

        # LDA vector arrays
        self.questionLDA = None # an ndarray of size (n_questions, n_topics) to hold the LDA vectors of the questions
        self.tags = None # an ndarray of size (n_questions, n_tags) to hold the boolean tag vectors for the questions

        # similarity matrices
        self.question_similarity = None
        self.tag_similarity = None
        #self.tfidf_similarity = None

        # latent factor vectors and indices
        self.lf_user = None
        self.lf_question = None
        self.lf_uidx = None
        self.lf_qidx = None


    # (re)train the recommender
    def train(self, num_topics = 100, iterations = 1000, passes = 1, df_dict = None, multicore = False, 
              cutoff = 4, compute_similarity = False, tfidf=False, train_LDA=True):
        """
        Builds the corpus from the supplied DataFrames, 
        creates a dictionary from it, and trains an LDA model
        and creates LDA vectors for each question in the corpus.

        Each user's answered questions are partitioned into 'good' (stars >= cutoff) 
        and 'bad' (stars < cutoff), and an average LDA+tag-vector is created for
        each class.
        
        Optional Parameters
        ===================
        num_topics : number of topics in in the LDA model
        iterations : number of iterations for training the LDA model
        passes : number of passes for training the LDA model
        multicore : (optional, default False) number of workers to use for multicore training (should be = cpu_count() - 1)
        cutoff : (optional, default 4) minimum stars for a 'good' question
        """

        #print "Training..."
        #sys.stdout.flush()

        if df_dict is not None:
            self.questions_df = df_dict['questions']
            self.answers_df = df_dict['answers']
            self.tags_df = df_dict['tags']
            self.users_df = df_dict['users']
            self.comments_df = df_dict['comments']
            self.fulldf = df_dict['fulldf']

        elif self.questions_df is None:
            print "Nothing to train. Exiting."
            return
        
        if train_LDA:
            corpus = pd.concat([self.questions_df.title+" "+self.questions_df.question,
                                self.answers_df.answer,
                                self.comments_df.comment])
            # strip the HTML - this will throw warnings if there are URLs in the corpus, but oh well
            corput = [BeautifulSoup(entry).get_text() for entry in corpus]
            site_corpus = [self.post_to_words(post) for post in corpus]
            
            # create the BoW dictionary
            self.dictionary = gensim.corpora.Dictionary(line for line in site_corpus)
            print " ...created the dictionary."
            
            # create the vectorized corpus
            self.vectorized_corpus = [self.dictionary.doc2bow(post) for post in site_corpus]
            print " ...vectorized the corpus."

            # tfidf transformation
            if tfidf:
                tfidf_model = gensim.models.TfidfModel(self.vectorized_corpus)
                self.vectorized_corpus = tfidf_model[self.vectorized_corpus]

            # train the LDA model
            print " ...training the LDA model on %d topics in %d passes and %d iterations." % (num_topics, passes, iterations)
            sys.stdout.flush()
            
            if not multicore:
                self.LDAmodel = gensim.models.LdaModel(self.vectorized_corpus, 
                                                       id2word=self.dictionary, 
                                                       num_topics = num_topics,
                                                       iterations = iterations, 
                                                       passes = passes)
            else: #this doesn't seem to work... 
                self.LDAmodel = gensim.models.ldamulticore.LdaMulticore(self.vectorized_corpus, 
                                                       id2word=self.dictionary, 
                                                       num_topics = num_topics,
                                                       iterations = iterations, 
                                                       passes = passes,
                                                       workers = multicore)
            print " ...trained the LDA model."
            sys.stdout.flush()

            # compute all the LDA and boolean-tag vectors

            # an ndarray of size (n_questions, n_topics) to hold the LDA vectors of the questions
            self.questionLDA = self.LDAvec2ndarray(self.post2LDAvec(self.questions_df.title + " " + self.questions_df.question))

            # an ndarray of size (n_questions, n_tags) to hold the boolean tag vectors for the questions
            self.tags = self.post2tagvec(self.questions_df.tags) 

        # compute amateur and expert vectors for each user

        if compute_similarity:
            self.compute_sim()
            print " ...computed similarity matrics."
            sys.stdout.flush()

        #print "Done training!"
        #sys.stdout.flush()

        return

    # (pre)compute the similarity matrices
    def compute_sim(self):
        """
        Compute the title, question and tags similarity matrices.
        """

        # check if the model is trained... if not, ask to train
        if self.LDAmodel == None:
            print "Model has not been trained yet."
            train_now = raw_input("Train (with default settings) or Load now? [t/l/N] ")
            if train_now == 't':
                self.train()
            if train_now == 'l':
                name = raw_input("Model to load: ")
                if name != None:
                    try:
                        self.load(name)
                    except:
                        print "No such model. Try training a model first. Exiting gracefully."
                        return
            else:
                return

        # questions and titles
        self.question_similarity = cosine_similarity(self.questionLDA, self.questionLDA)
        self.tag_similarity = np.dot(self.tags, self.tags.T)

        return
    
    # save a trained model    
    def save(self, name):
        """
        Save a trained model to disk. The dictionary, vectorized corpus,
        and LDA model will be saved (resp.) to 
        
        name + "_dict.pkl"
        name + "_vectCorpus.mm"
        name + "_LDAmodel.pkl"
        """

        print "Saving the model..."
        
        if self.LDAmodel == None:
            print "Model hasn't been trained yet."
            return

        try:
            self.dictionary.save('data/'+self.site_name+'/'+name+'_dict')
            print " ...saved the dictionary."
            sys.stdout.flush()

            gensim.corpora.MmCorpus.serialize('data/'+self.site_name+'/'+name+'_vectCorpus.mm', 
                                              self.vectorized_corpus)
            print " ...saved the corpus."
            sys.stdout.flush()

            self.LDAmodel.save('data/'+self.site_name+'/'+name+'_LDAmodel')
            print " ...saved the LDA model."
            sys.stdout.flush()

            pkl.dump(self.questionLDA, gzip.GzipFile('data/'+self.site_name+'/'+name+'_questionLDA', 'wb'))
            pkl.dump(self.tags, gzip.GzipFile('data/'+self.site_name+'/'+name+'_tagVec', 'wb'))
            print " ...saved the LDA vectors."
            sys.stdout.flush()

            if self.question_similarity is not None:
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
            self.dictionary = gensim.corpora.Dictionary.load('data/'+self.site_name+'/'+name+'_dict')
            self.vectorized_corpus = gensim.corpora.MmCorpus('data/'+self.site_name+'/'+name+'_vectCorpus.mm')
            self.LDAmodel = gensim.models.LdaModel.load('data/'+self.site_name+'/'+name+'_LDAmodel')
            print " ...LDA model loaded."
            sys.stdout.flush()

            """
            self.question_similarity = pkl.load(gzip.GzipFile('data/'+self.site_name+'/'+name+'_questionSim', 'rb'))
            print " ...question similarities loaded."
            sys.stdout.flush()

            self.tag_similarity = pkl.load(gzip.GzipFile('data/'+self.site_name+'/'+name+'_tagSim', 'rb'))
            print " ...tag similarities loaded."
            sys.stdout.flush()
            """

            self.questionLDA = pkl.load(gzip.GzipFile('data/'+self.site_name+'/'+name+'_questionLDA', 'rb'))
            print " ...question LDA vectors loaded."
            sys.stdout.flush()

            self.tags = pkl.load(gzip.GzipFile('data/'+self.site_name+'/'+name+'_tagVec', 'rb'))
            print " ...tag vectors loaded."
            sys.stdout.flush()

            self.lf_question = pkl.load(gzip.GzipFile('data/'+self.site_name+'/pfeatures.gzpkl', 'rb'))
            self.lf_user = pkl.load(gzip.GzipFile('data/'+self.site_name+'/ufeatures.gzpkl', 'rb'))
            self.lf_uidx = pkl.load(open('data/'+self.site_name+'/uidx.pkl', 'rb'))
            self.lf_qidx = pkl.load(open('data/'+self.site_name+'/midx.pkl', 'rb'))

            print "Successfully loaded the recommender."
            sys.stdout.flush()

        except:
            print "Error loading the model."
            
        return
    
    
    # helper function to convert post text (question, answer, comment, title)
    #   to a list of words

    def post_to_words(self, raw_post):
        """Convert the output of BeautifulSoup.getText() to a list 
        of loweselfase words. Remove punctuation, urls, and single letters.
        """

        #split on white space and remove urls
        words = raw_post.split()
        words = ' '.join([w for w in words if w[:4] != 'http'])

        # remove punctuation, convert to lower case 
        words = re.sub("[^a-zA-Z]"," ", words).lower().split()

        # remove stopwords and single characters
        stops = stopwords.words("english")
        letters = list(string.ascii_lowercase)
        stops.extend(letters)
        stops = set(stops)

        words = [w for w in words if w not in stops]

        return words
    
    
    # some functions to convert posts -> LDAvec -> ndarray
    def LDAvec2ndarray(self, LDAvec):
        """
        Converts the (apparently) sparse LDA-vectors to an ndarray. 

        Inputs
        ======
        LDAvec : LDAvec or numpy array of LDA vectors
        """

        vec = np.array(LDAvec)
        if isinstance(LDAvec, list): # if it is a list, convert to an ndarray with .shape object
            output = np.zeros(self.LDAmodel.num_topics)
            for x in vec:
                output[int(x[0])] = x[1]
            return output
        else:
            output = np.zeros((vec.shape[0],self.LDAmodel.num_topics))
            for j in range(vec.shape[0]):
                for x in vec[j]:
                    output[j,int(x[0])] = x[1]
            return output

    def post2LDAvec(self, question):
        """
        Converts post (BeautifulSoup.getText() output) to an LDAvec.
        """

        if isinstance(question, str): # got a single question/title
            return self.LDAmodel[self.dictionary.doc2bow(self.post_to_words(question))]
        else: # got a Series
            return np.array([self.LDAmodel[self.dictionary.doc2bow(self.post_to_words(text))] for text in question])      

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

    # cosine similarity between LDA vectors
    def LDAsim(self, x, y):
        """
        Compute the cosine similarity between the given pd.Series of LDA-vectors.
        Inputs
        ======
        x : LDA vector
        y : LDA vector

        Since LDA vectors are always nonnegative, the returned cosine similarity will be in [0,1]."""

        X = self.LDAvec2ndarray(x)
        Y = self.LDAvec2ndarray(y)

        return cosine_similarity(X, Y)[0]


    # aggregated similarity between questions (title+question+tags)
    def post_sim(self, q1, q2, new1 = False, new2 = False, w = [1,1,1]):
        """Use for computing similarities of new posts. Not yet implemented."""
        return

    
    # k-NN for the questions based on weighted cosine + boolean tag similarity
    def k_nearest_questions(self, question_id, qdf, k = 7, w = [1,1,1], similarity = 'LDA'):
        """
        Returns the k nearest neighbors of the question among those questions in questions_df.

        Inputs
        ======
        question : an index of self.question_df
        qdf : a subindex of self.questions_df.index
        k : (optional) number of neighbors
        w : (optional) weights used in the similarity metric (title, question, tags)
        similarity : (str, optional) similarity matrix to use. Currently supported: 
                                        'LDA' : cosine similarity of LDA vectors
                                        'tag' : (dot product of boolean tag vectors)
        """

        if similarity == 'tag':
            sim_matrix = self.tag_similarity
        elif similarity == 'LDA':
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
                 'LF+LDA' : LF model + LDA-similarity-based collaborative filter (CF) for the residues (not yet implemented)
                 'LF+tag' : LF model + tag-similarity based CF for the residues (NYI)
                 'LF+Tfidf' : LF model + tfidf-3gram-similarity based CF for the residues (NYI)

        Returns
        =======
        (float) the user_id's predicted stars for answering the given question

        If the user has answered the question, returns the user's answer stars."""

        if method == "CF":
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

            # make a list of the question means indexed by question_id
            qmeans = self.fulldf.groupby('question_id').item_mean.mean()

            prediction = normed_stars + self.fulldf[self.fulldf.user_id == user_id].user_mean.irow(0) \
                                      + qmeans.ix[question_id] \
                                      - self.fulldf.stars.mean()

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
                 'LF+LDA' : LF model + LDA-similarity-based collaborative filter (CF) for the residues (not yet implemented)
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

    # make recommendations for all uses and store the list of recommended question_id's in self.user_df
    def recommend_all(self, N = 10, method = 'LF', save = True):
        
        def rq(uid):
            if uid == '-999':
                return None
            else:
                return self.recommend_questions(uid, N = N, method = method)
        
        self.users_df['recommendations'] = self.users_df.index.map(rq)

        if save:
            dfs = {'questions':self.questions_df,
                   'answers':self.answers_df,
                   'tags':self.tags_df,
                   'users':self.users_df,
                   'comments':self.comments_df}

            with gzip.GzipFile('data/'+self.site_name+'/dataframes.gzpkl', 'wb') as f:
                pkl.dump(dfs, f)

            with gzip.GzipFile('data/'+self.site_name+'/userdf.gzpkl', 'wb') as f:
                pkl.dump(self.users_df, f)

            with gzip.GzipFile('data/'+self.site_name+'/qdf.gzpkl', 'wb') as f:
                pkl.dump(self.questions_df, f)
            
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
        Returns the top N most similar questions (using LDA vectors).
        """
        
        LDAvec = self.post2LDAvec(title+" "+question)
        qvec = self.LDAvec2ndarray(LDAvec)
        
        similarities = pd.Series(cosine_similarity(qvec,self.questionLDA)[0], index=self.question_idx.index)
        similarities.sort(ascending=False)
        
        return similarities.head(N).index