import nltk
import pandas as pd
import numpy as np
import heapq
from sklearn import metrics
from collections import Counter
from textblob import TextBlob, Word
from nltk.corpus import stopwords
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import SparsePCA
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, auc, classification_report, confusion_matrix, \
    f1_score, make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, KFold, train_test_split, \
    RepeatedKFold
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline as Pipe
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

'''
clean_data: drops NaN values and isolates wanted columns from the given JSON file
input: data       Given training JSON file
output: data      Cleaned dataframe
'''
def clean_data(data):
    # Remove NaN values from the dataset
    data1 = data.dropna(how='all')
    # Drop unwanted columns
    data2 = data1.drop(['image', 'style', 'vote', 'unixReviewTime'], axis=1)
    data3 = data2.dropna(subset=['summary', 'reviewText'])
    data4 = data3.drop(['verified'], axis=1)
    data_clean = data4.drop(['reviewerID', 'reviewTime', 'reviewerName'], axis=1)
    return data_clean


'''
preprocess_data: groups the data by product (asin)
input: data       Cleaned dataframe from clean_data
output: data      Processed dataframe
'''
def preprocess_data(df):
    col = ['asin']
    # Find the average of the overall rating for each product
    df_concate = df.groupby(col)[["overall"]].mean().add_suffix('_Mean Ratings')
    # Combine the text for multiple product rows into one
    df_update = df.join(df_concate, on=col)
    grouped_df = df_update.groupby(['asin'], as_index=False).agg(
        lambda x: x.mean() if x.dtype == 'float64' else '. '.join(x))
    return grouped_df


'''
label: puts data into Awesome or Not Awesome based on ratings
input: row       product row
output: label    class label
'''
def label(row):
    if row['overall_Mean Ratings'] > 4.70:
        label = "1"
    else:
        label = "0"
    return label


"""
sentiment_extraction: extract sentiment polarity from reviewText and summary
input: data			cleaned data frame
input: output		name of csv file to output to
output: output		csv file with new columns
"""
def sentiment_extraction(data, output):
    review_sentiment = []
    summary_sentiment = []

    # Loop through the reviewText and summary columns
    for (review, summary) in zip(data['reviewText'], data['summary']):
        # Extrace the sentiment scores
        r_score = TextBlob(review)
        s_score = TextBlob(summary)
        review_sentiment.append(r_score.sentiment.polarity)
        summary_sentiment.append(s_score.sentiment.polarity)

    data['summarySentiment'] = summary_sentiment
    data['reviewSentiment'] = review_sentiment

    # Write out dataset with new columns to a csv
    data.to_csv(output)


"""
subjectivity_extraction: extract sentiment subjectivity from reviewText and summary
input: data			cleaned data frame
input: output		name of csv file to output to
output: output		csv file with new columns
"""
def subjectivity_extraction(data, output):
    review_sentiment = []
    summary_sentiment = []

    # Loop through the reviewText and summary columns
    for (review, summary) in zip(data['reviewText'], data['summary']):
        # Extract the sentiment subjectivity
        r_score = TextBlob(review)
        s_score = TextBlob(summary)
        review_sentiment.append(r_score.sentiment.subjectivity)
        summary_sentiment.append(s_score.sentiment.subjectivity)

    data['summarySub'] = summary_sentiment
    data['reviewSub'] = review_sentiment

    # Write out dataset with new columns to a csv
    data.to_csv(output)


'''
clean_text: tokenizes, lemmatizes, and removes stop words and punctuation from reviews
input: data        dataframe with classes
output: data       dataframe with individual words in review
'''
def clean_text(data):
    tk = nltk.RegexpTokenizer(r"\w+")  # remove punctuation
    stop_words = set(stopwords.words('english'))

    # First, the reviewText column
    new_reviews = []
    for review in data['reviewText']:
        review = tk.tokenize(review)
        output = []
        for word in [w for w in review if not w in stop_words]:
            output.append(word)
        new_reviews.append(' '.join(review).lower())

    # Then clean the summary column
    new_summaries = []
    for summary in data["summary"]:
        summary = tk.tokenize(summary)
        output = []
        for word in [w for w in summary if not w in stop_words]:
            output.append(word)
        new_summaries.append(' '.join(output).lower())

    data['reviewText'] = new_reviews
    data['summary'] = new_summaries

    return data


"""
generate_under_over_sample: Geneartes random undersample and oversampled training data
input: data        unbalanced data set
output: data       seperate unbalanced test data, undersampled and oversampled training data
"""
def generate_under_over_sample(data):
    df_train = data
    # Randomly Split 70-30 and keep 30% as Test data
    df_test_resample = df_train.sample(frac=0.3)
    # Take the 70% data as Training
    df_under_train = pd.concat([df_test_resample, df_train]).drop_duplicates(keep=False)
    # Take Awesome class samples in one dataframe and not awesome in the other
    df_train_not_awesome = df_under_train[df_under_train['label'] == 0]
    df_train_awesome = df_under_train[df_under_train['label'] == 1]
    # Count "1" and "0" class counts in the Training Data
    count_not_awesome, count_awesome = df_under_train.label.value_counts()
    # Undersampling not awesome class to make same count as awesome class
    df_train_not_awesome_under = df_train_not_awesome.sample(count_awesome)
    # Concate the undersampled not awesome with the awesome class (makes equal samples)
    df_resample = pd.concat([df_train_not_awesome_under, df_train_awesome], axis=0)
    # Export the undersampled data frame as csv
    df_resample.to_csv('Training_Undersample.csv', index=False)
    # Save 30% unbalanced data as test data
    df_test_resample.to_csv('Testing.csv', index=False)
    # Oversampling awesome class to make same count as not awesome class
    df_train_awesome_over = df_train_awesome.sample(count_not_awesome, replace=True)
    # Concate the Oversampled awesome with the not awesome class (makes equal samples)
    df_resample2 = pd.concat([df_train_awesome_over, df_train_not_awesome], axis=0)
    # Export the Oversampled data frame as csv
    df_resample2.to_csv('Training_Oversample.csv', index=False)


"""
Class Selector --
Idea for this class came from online research (https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines) 
to get Pipelines to work properly.
Returns the vector from all feature vectors corresponding to its key when transformed
"""
class Selector:
    def __init__(self, key):
        self.key = key

    def fit(self, feat, labels=None):
        return self

    def transform(self, feat):
        return feat[self.key]


"""
grid: perform Hyperparameter Optimization via GridSearchCV
input: data		dataset of your choosing
"""
def grid(data):
    # Put the features into a numpy array
    features = np.array(['reviewText', 'summary', 'reviewSentiment', 'summarySentiment', 'reviewSub', 'summarySub'])

    # Use count vectorizer to extract features from TF-IDF
    revTf = Pipeline(steps=[('selector', Selector(key='reviewText')), ('cf', CountVectorizer(max_df=.95)),
                            ('tf', TfidfTransformer(sublinear_tf=True))])
    sumTf = Pipeline(steps=[('selector', Selector(key='summary')), ('cf', CountVectorizer(max_df=.95)),
                            ('tf', TfidfTransformer(sublinear_tf=True))])

    # Normalize all the features
    combine = ColumnTransformer(transformers=[('revSen', StandardScaler(), ['reviewSentiment']),
                                              ('sumSen', StandardScaler(), ['summarySentiment']),
                                              ('revSub', StandardScaler(), ['reviewSub']),
                                              ('sumSub', StandardScaler(), ['summarySub']),
                                              ('rev', revTf, ['reviewText']), ('sum', sumTf, ['summary'])])
    # Apply the model
    model = Pipeline(steps=[('combine', combine), ('classifier', LogisticRegression())])
    scorer = make_scorer(f1_score)

    # Set up the grid, defining parameter to check
    grid = GridSearchCV(model, param_grid={'classifier__max_iter': [500],
                                           'classifier__C': [.8], 'combine__rev__cf__min_df': [.01, .4, .1]},
                        scoring=scorer, cv=10, n_jobs=-1,
                        verbose=True)

    # Fit the grid
    print("Fitting Grid")
    feat_train, feat_test, labels_train, labels_test = train_test_split(data[features], data["label"], test_size=0.3)
    grid.fit(feat_train, labels_train)
    print("Best parameters (F1 =%0.3f):" % grid.best_score_)
    print(grid.best_params_)

    # Test features on partitioned data, as a sanity check that we haven't overfit
    preds = grid.predict(feat_test)
    print(metrics.confusion_matrix(labels_test, preds))
    print(classification_report(labels_test, preds))


"""
majority_smote: majority voting model using SMOTE oversampled Data
input: data		unbalanced dataset
Input: clf		classifier to apply
"""
def majority_smote(data):
    df_train = data
    # Make 70-30 Split.
    # Use 30% as test data and use SMOTE on 70% training data to balance it
    df_test_resampled = data.sample(frac=0.3)
    df_train_resampled = pd.concat([df_test_resampled, df_train]).drop_duplicates(keep=False)

    # Extract TF-IDF
    vect = TfidfVectorizer(sublinear_tf=True, max_df=0.98, analyzer='word', stop_words='english')

    # Split and fit the data
    X_train = vect.fit_transform(df_train_resampled.pop('reviewText'))
    X_test = vect.transform(df_test_resampled.pop('reviewText'))
    y_train = df_train_resampled[['label']]
    y_test = df_test_resampled[['label']]

    sm = SMOTE(k_neighbors=2, random_state=2)
    # SMOTE fiting Training Data
    X_train, y_train = sm.fit_sample(X_train, y_train)
    ##10 fold cv 3 repeats
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=500)

    # best modles with best hyperparameters
    model1 = LinearSVC()
    model2 = LogisticRegression(penalty='l2', max_iter=1000, C=1)
    model3 = RandomForestClassifier(min_samples_split=3, max_depth=10, n_estimators=1000, n_jobs=-1, random_state=500)

    # Fitting models
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    model3.fit(X_train, y_train)

    ##Obtain Predictions
    yhat1 = model1.predict(X_test)
    yhat2 = model2.predict(X_test)
    yhat3 = model3.predict(X_test)

    ##Model Scores
    final_score1 = cross_val_score(model1, X_test, y_test, scoring='f1', cv=cv, n_jobs=-1)
    final_score2 = cross_val_score(model2, X_test, y_test, scoring='f1', cv=cv, n_jobs=-1)
    final_score3 = cross_val_score(model3, X_test, y_test, scoring='f1', cv=cv, n_jobs=-1)

    ##Build the Majority Voting Classifier
    ##Considering Equal Weights for each three classifiers above
    yhat = (yhat1 + yhat2 + yhat3) / 3.00
    for i in range(yhat.shape[0]):
        if yhat[i] > 0.5:
            yhat[i] = 1.0
        else:
            yhat[i] = 0.0

    # report performance
    print(confusion_matrix(y_test, yhat.round()))
    print(classification_report(y_test, yhat.round()))
    print(accuracy_score(y_test, yhat.round()))


"""
vote_mode: model using soft voting ensemble
input: data 		training data
input: test 		unbalanced test data
"""
def vote_model(data, test):
    # Create a numpy array of features
    features = np.array([c for c in data.columns.values if
                         c in ['reviewText', 'summary', 'reviewSentiment', 'summarySentiment', 'reviewSub',
                               'summarySub']])

    # set up pipelines to run tf-idf
    revTf = Pipeline(steps=[('selector', Selector(key='reviewText')), ('cf', CountVectorizer(max_df=.9)),
                            ('tf', TfidfTransformer(sublinear_tf=True))])
    sumTf = Pipeline(steps=[('selector', Selector(key='summary')), ('cf', CountVectorizer(max_df=.9)),
                            ('tf', TfidfTransformer(sublinear_tf=True))])

    # run it on ten folds
    for i in range(0, 10):
        print("Fold #", i + 1)
        feat_train, feat_test, labels_train, labels_test = train_test_split(data[features], data["label"],
                                                                            test_size=0.3)

        # set up column transformers
        combine = ColumnTransformer(transformers=[('revSen', StandardScaler(), ['reviewSentiment']),
                                                  ('sumSen', StandardScaler(), ['summarySentiment']),
                                                  ('revSub', StandardScaler(), ['reviewSub']),
                                                  ('sumSub', StandardScaler(), ['summarySub']),
                                                  ('rev', revTf, ['reviewText']), ('sum', sumTf, ['summary'])])

        # five different models
        model1 = Pipeline(
            steps=[('combine', combine), ('classifier', SVC(kernel='linear', probability=True, max_iter=500, C=.8))])
        model2 = Pipeline(
            steps=[('combine', combine), ('classifier', SVC(kernel='linear', probability=True, max_iter=1000, C=.6))])
        model3 = Pipeline(
            steps=[('combine', combine), ('classifier', SVC(kernel='linear', probability=True, max_iter=500, C=.6))])
        model4 = Pipeline(
            steps=[('combine', combine), ('classifier', SVC(kernel='linear', probability=True, max_iter=1000, C=.8))])
        model5 = Pipeline(steps=[('combine', combine),
                                 ('classifier', LogisticRegression(max_iter=1000, warm_start=True, penalty='l2'))])
        model6 = Pipeline(steps=[('combine', combine),
                                 ('classifier', LogisticRegression(max_iter=2000, warm_start=True, penalty='l2'))])

        # put models into the ensemble
        models = [('svm1', model1), ('svm2', model2), ('svm3', model3), ('svm4', model1), ('LR1', model5),
                  ('LR2', model6)]
        ensemble = VotingClassifier(estimators=models, voting="soft")

        print("Fitting")

        # Fit the model
        ensemble.fit(feat_train, labels_train)
        print("Model fit")

        # gather predictions on the test data
        preds = ensemble.predict(test[features])

        # show results
        print(metrics.confusion_matrix(test['label'], preds))
        print(classification_report(test['label'], preds))


"""
smote: test different classifiers using SMOTE overbalancing -- requires use of imblearn Pipeline
input: data		unbalanced dataset
input: clf		classifier to apply
"""
def smote(data, clf):
    """
    Run a model with 10 fold cv using SMOTE overbalancing
    """
    features = np.array(['reviewText', 'summary', 'reviewSentiment', 'summarySentiment', 'reviewSub', 'summarySub'])
    # set up pipelines to run tf-idf
    revTf = Pipe(steps=[('selector', Selector(key='reviewText')), ('cf', CountVectorizer(max_df=.95)),
                        ('tf', TfidfTransformer(sublinear_tf=True))])
    sumTf = Pipe(steps=[('selector', Selector(key='summary')), ('cf', CountVectorizer(max_df=.95)),
                        ('tf', TfidfTransformer(sublinear_tf=True))])
    overallf1avg = 0
    awesomef1avg = 0
    # fold ten times
    for i in range(0, 10):
        print("Fold #", i + 1)
        # random 70-30 split
        feat_train, feat_test, labels_train, labels_test = train_test_split(data[features], data["label"],
                                                                            test_size=0.3)
        # set up column transformers to put all features together
        combine = ColumnTransformer(transformers=[('revSen', StandardScaler(), ['reviewSentiment']),
                                                  ('sumSen', StandardScaler(), ['summarySentiment']),
                                                  ('revSub', StandardScaler(), ['reviewSub']),
                                                  ('sumSub', StandardScaler(), ['summarySub']),
                                                  ('rev', revTf, ['reviewText']), ('sum', sumTf, ['summary'])])

        # put the final model pipeline together
        model = Pipeline(steps=[('combine', combine), ('classifier', clf)])

        # fit the model
        print("Fitting")
        model.fit(feat_train, labels_train)

        # make predictions and display results
        print("Model fit")
        preds = model.predict(feat_test)
        print(metrics.confusion_matrix(labels_test, preds))
        print(classification_report(labels_test, preds))
        overallf1avg += f1_score(labels_test, preds, average='weighted')
        awesomef1avg += f1_score(labels_test, preds, average='binary')

    # Calculate the average overall and awesome f1s
    print("Overall F1: ", overallf1avg / 10)
    print("Awesome F1: ", awesomef1avg / 10)


"""
output_test_smote: output a test file using SMOTE overbalancing
input: train		training dataset
input: test			testing dataset
"""
def output_test_smote(train, test):
    features = np.array(['reviewText', 'summary', 'reviewSentiment', 'summarySentiment', 'reviewSub', 'summarySub'])
    # set up pipeline transformers
    revTf = Pipe(steps=[('selector', Selector(key='reviewText')), ('cf', CountVectorizer(max_df=.95)),
                        ('tf', TfidfTransformer(sublinear_tf=True))])
    sumTf = Pipe(steps=[('selector', Selector(key='summary')), ('cf', CountVectorizer(max_df=.95)),
                        ('tf', TfidfTransformer(sublinear_tf=True))])

    # set up column transformers
    combine = ColumnTransformer(transformers=[('revSen', StandardScaler(), ['reviewSentiment']),
                                              ('sumSen', StandardScaler(), ['summarySentiment']),
                                              ('revSub', StandardScaler(), ['reviewSub']),
                                              ('sumSub', StandardScaler(), ['summarySub']),
                                              ('rev', revTf, ['reviewText']), ('sum', sumTf, ['summary'])])

    # set up final model pipeline
    model = Pipe(steps=[('combine', combine), ('smote', SMOTE()),
                        ('classifier', LogisticRegression(max_iter=500, warm_start=True, penalty='l2', C=.8))])

    # Fit the data and predict
    print("Fitting")
    model.fit(train[features], train['label'])
    print("Model fit")
    preds = model.predict(test[features])

    # Output to an output file
    df = pd.DataFrame(np.dstack((test["asin"], preds))[0], columns=["asin", "label"])
    df.to_csv("output_smote.csv")


'''
main: runs all functions
'''
def main():
    # Read in the JSON training data
    df = pd.read_json('Automotive_Reviews_training.json', lines=True)
    # Clean the dataframe to isolate the data needed for features
    df = clean_data(df)
    df = preprocess_data(df)
    # Apply class labels
    df['label'] = df.apply(label, axis=1)
    # Extract the sentiment scores from text for the training data
    sentiment_extraction(df, "PreprocessedAutomotive.csv")
    subjectivity_extraction(df, "PreprocessedAutomotive.csv")

    # Read in the JSON training data
    test = pd.read_json('Automotive_Reviews_test.json', lines=True)
    # Clean the dataframe to isolate the data needed for features
    test = clean_data(test)
    test = preprocess_data(test)
    # Extract sentiment scores from the text for the test data
    sentiment_extraction(df, "Automotive Test.csv")
    subjectivity_extraction(df, "Automotive Test.csv")

    # Read in both clean files
    df = pd.read_csv("PreprocessedAutomotive.csv")
    test = pd.read_csv("Automotive Test.csv")

    # Generate the output file with Logistic Regression, our classifier of choice
    output_test_smote(df, test)


main()
