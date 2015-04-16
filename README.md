# Stack-Exchange-Recommender

A Recommender System for the Stack Exchange family of websites. The system is built on the periodic data dump of the Stack Exchange family of websites available at the [Internet Archive ](https://archive.org/details/stackexchange). 

Currently, recommendations are made only for users who have answered at least one question. The recommendations are based on a bias-corrected latent factor model (a so-called SVD model in the terminoligy of the Netflix papers). 

The recommender system also trains an LDA (latent Dirichlet allocation) topic model for comparing new questions/answers to old questions/answers, although this feature is not yet implemented. There is also lots of zombie code which will model the residuals of the latent factor model with a collaborative filter based on various types of similarity (cosine similarity of LDA vectors, Tf-idf n-grams, Jaccard similarity of boolean tag vectors, etc...).

## Dependences

The system is built in Python 2.7 with the usual scientific stack (numpy, pandas, scikit-learn), and has the following less standard dependencies:

- The LDA model uses Radim Řehůřek's wonderful [gensim](https://radimrehurek.com/gensim/index.html) library.
- The SVD model is built and trained with Apache (py)Spark, although only local contexts are supported at this point.

## Usage

There are currently 270 sites in the Stack Exchange family; see `site_names.csv` for a full list. The steps to build recommendations for <site_name> are as follows:

1. run `python makesite.py <site_name>` . This will scrape the data from the internet archive and parse it into pandas dataframes in the local subdirectory `data/<site_name>`.

2. run `python makeLF.py <site_name>` to train the LDA model (for features which haven't been implemented yet) and the SVD model. This will also save the models to the directory `data/<site_name>`.

3. run `python makeRecommendations.py <site_name>` to make 10 recommendations for all users.

4. Recommendations are available in the pandas dataframe `users_df` pickled in `data/<site_name>/dataframes.gzpkl`.

The main class for dealing with the scraping and parsing of the sites is StackSite (in `stacksite.py`), and the main Recommender class, which includes various methods for making and retrieving recommendations as well as training various components of the models, is contained in `stackrecommender.py`.

## Notes

There are a lot of features missing: closed questions are included in the recommendations, age of questions is not yet accounted for, and all features involving new questions (and hence based on LDA decompositions) are still missing.

The system will also probably crash on any of the larger sites for various reasons (e.g., my Spark implementation is still primitive and runs into memory problems, the file structure of StackOverflow is nonstandard, etc...).
