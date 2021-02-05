import pandas as pd
import numpy as np
import pickle, json, requests
import matplotlib.pyplot as plt
import re,os,glob,string,itertools
import math,tqdm,time,operator
from numpy import save, load
from nltk.tokenize import word_tokenize,sent_tokenize
from flask import Flask, session, flash, redirect, render_template, request, url_for
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word
from bs4 import BeautifulSoup
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from string import punctuation
from keras.models import Sequential
from keras.layers import Input,Dense,Embedding,LSTM,Dropout,Bidirectional,SpatialDropout1D,BatchNormalization
from keras.preprocessing import text, sequence
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import regularizers
from gensim.models import KeyedVectors # load the Stanford GloVe model
from keras.regularizers import l2
from keras import optimizers
from keras.callbacks import EarlyStopping
from multiprocessing import Process
import seaborn as sns
sns.set(color_codes=True)
sns.set_theme(style="darkgrid")
import xgboost as xgb
from xgboost import XGBClassifier
import sklearn
import sklearn.ensemble
from sklearn.preprocessing import MinMaxScaler, binarize
from collections import defaultdict
from collections import  Counter
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)
app.secret_key = 'random string'
output=[]#("message stark","hi")]
images = 'C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\images\\'


@app.route('/')
def index():
    return render_template('login.html')

@app.route('/options', methods=['GET', 'POST'])
def options():
    if request.method == 'POST':
        result = request.form
        if request.form['submit_button'] == "Preprocess":
            return redirect(url_for('preprocess'))
        elif request.form['submit_button'] == "Trainclassifiers":
            global p1, p2, p3, p4, p5, p6, p7
            # training_DL_data_preparation()
            p1 = Process(target=training_DL_data_preparation)
            p2 = Process(target=training_ML_data_preparation)
            p1.start()
            p2.start()
            p1.join()
            p2.join()
            p3 = Process(target=train_bidirectional_lstm)
            p4 = Process(target=train_svm_classifier)
            p5 = Process(target=train_XGBOOST_classifier)
            p6 = Process(target=train_randomforest_classifier)
            p7 = Process(target=train_ann_classifier)
            p3.start()
            p4.start()
            p5.start()
            p6.start()
            p7.start()
            flash('Training Initiated')
            print("=== Training Initiated ===")
            return redirect(url_for('options'))
        elif request.form['submit_button'] == "ChatBot":
            return render_template("chat_page.html",result=output)
        else:
            pass  # unknown
    elif request.method == 'GET':
        return render_template('options.html')
    return render_template('options.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    username = request.form["username"]
    pwd = request.form["password"]
    user_df = read_usercsv()
    auth_code_returned, user_role_returned = authenticate_user(user_df,username,pwd)

    if request.method == 'POST':
        if not auth_code_returned:
            flash('Invalid username or pwd. Pls try again!')
            return render_template('login.html')
        else:
            flash('You were successfully logged in')
            return redirect(url_for('options'))

    return render_template('login.html', error=error)

@app.route('/result',methods=["POST","GET"])
def Result():
    if request.method=="POST":
        result=list(request.form.values())[0]
        if result.lower()=="restart":
            output.clear()
        else:
            try:
                data = json.dumps({"sender": "Rasa", "message": result})
                # r = requests.post('http://localhost:5002/webhooks/rest/webhook', json={"message": result})
                headers = {'Content-type': 'application/json'}
                r = requests.post('http://localhost:5002/webhooks/rest/webhook', data= data, headers= headers)
                print("Bot says, ")
                temp=[]
                for i in r.json():
                    bot_message = i['text']
                    temp.extend([bot_message])
                    print(f"{i['text']}")
                output.extend([("message parker",result),("message stark",bot_message)])
                # output.extend([("message parker",result),("message stark",temp)])

            except:
                output.extend([("message parker", result), ("message stark", "We are unable to process your request at the moment. Please try again...")])

        return render_template("chat_page.html",result=output)


@app.route('/preprocess',methods=['GET', 'POST'])
def preprocess():
    flask = "Y"
    path = 'C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\preprocessing'
    train_path = 'C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\training'

    if request.method == 'POST':
        result = request.form
        if request.form['submit_button'] == "InitiateP":
           df, rep_df = read_inputfiles(flask, path)
           df = _data_cleansing(df)
           df.reset_index(inplace=True)

           df =  date2YMDW(df)

           print("\nConvert month variable into seasons:")
           df['Season'] = df['Month'].apply(month2seasons)
           print("\nConvert month variable into quarters:")
           df['Quarter'] = df['Month'].apply(month2quarter)
           df.rename(columns={'Genre': 'Gender', 'Employee or Third Party': 'Employee type'}, inplace=True)
           uniandmultivariate_analysis(df)
           wordcloud_nlpanalysis(df)
           # retained_cols = ["Description", "Description_denoised", "Accident Level", "Potential Accident Level"]
           retained_cols = ["Industry Sector", "Gender", "Employee type", "Critical Risk", "Description_denoised", "Accident Level"]
           df_training = df[retained_cols]
           # df_training.rename(columns={'Genre': 'Gender', 'Employee or Third Party': 'Employee type'}, inplace=True)
           df_training.to_csv(train_path + '\\' + 'prepro_data.csv')
           flash('Preprocessing successfully completed')
           return redirect(url_for('options'))

    elif request.method == 'GET':
        return render_template('preprocess.html', tree=make_tree(path))

    return render_template('preprocess.html', tree=make_tree(path))
    # return

def save_image(image_name):
    plt.savefig(images + '\\' + image_name + ".png")

def save_plotly_image(fig,image_name):
    fig.write_image(images + '\\' + image_name + ".png")


def wordcloud_nlpanalysis(df):
    # Number of characters present in each sentence.
    print("\nNumber of characters present in each sentence: ")
    df['Description'].str.len().hist()
    # plt.show()
    save_image("wordcloud_char")
    # Number of words appearing in each description
    print("\nNumber of words appearing in each description: ")
    df['Description'].str.split().map(lambda x: len(x)).hist()
    # plt.show()
    save_image("wordcloud_words")
    # Average word length
    print("\nAverage word length: ")
    df['Description'].str.split().apply(lambda x: [len(i) for i in x]).map(lambda x: np.mean(x)).hist()
    # plt.show()
    save_image("wordcloud_avgwordlen")
    # Stop word analysis
    print("\nTop Stop words analysis")
    plot_top_stopwords_barchart(df['Description'])
    # Top non-stop words plot
    print("\nTop non-stop words plot")
    plot_top_non_stopwords_barchart(df['Description'])
    # Analyzing N-Grams stop words
    print("\nAnalyzing N-Grams stop words")
    plot_top_ngrams_barchart(df['Description'], 2)
    plot_top_ngrams_barchart(df['Description'], 3)
    plot_top_ngrams_barchart(df['Description'], 4)
    # Plot word cloud
    print("\nWord Cloud")
    plot_wordcloud(df['Description'])
    return

def plot_top_stopwords_barchart(text):
    nltk.download('stopwords')
    stop = set(stopwords.words('english'))
    new = text.str.split()
    new = new.values.tolist()
    corpus=[word for i in new for word in i]
    dic=defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word]+=1
    top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]
    x,y=zip(*top)
    plt.bar(x,y)
    # plt.show()
    save_image("topstopwords")

def plot_top_non_stopwords_barchart(text):
    nltk.download('stopwords')
    stop = set(stopwords.words('english'))
    new = text.str.split()
    new = new.values.tolist()
    corpus = [word for i in new for word in i]
    counter = Counter(corpus)
    most = counter.most_common()
    x, y = [], []
    for word, count in most[:40]:
        if (word not in stop):
            x.append(word)
            y.append(count)
    sns.barplot(x=y, y=x)
    # plt.show()
    save_image("topnonstopwords")


####################################################################################################################################

# Analyzing N-Grams
from sklearn.feature_extraction.text import CountVectorizer


def plot_top_ngrams_barchart(text, n=2):
    stop = set(stopwords.words('english'))
    new = text.str.split()
    new = new.values.tolist()
    corpus = [word for i in new for word in i]

    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:10]

    top_n_bigrams = _get_top_ngram(text, n)[:10]
    x, y = map(list, zip(*top_n_bigrams))
    sns.barplot(x=y, y=x)
    # plt.show()
    save_image("topngrams")


####################################################################################################################################

# Plot word cloud
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
import nltk

nltk.download('punkt')
nltk.download('wordnet')


def plot_wordcloud(text):
    nltk.download('stopwords')
    stop = set(stopwords.words('english'))

    def _preprocess_text(text):
        corpus = []
        stem = PorterStemmer()
        lem = WordNetLemmatizer()
        for news in text:
            words = [w for w in word_tokenize(news) if (w not in stop)]

            words = [lem.lemmatize(w) for w in words if len(w) > 2]

            corpus.append(words)
        return corpus

    corpus = _preprocess_text(text)

    wordcloud = WordCloud(
        background_color='white',
        stopwords=set(STOPWORDS),
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)

    wordcloud = wordcloud.generate(str(corpus))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    plt.imshow(wordcloud)
    # plt.show()
    save_image("plot_wordcloud")


def uniandmultivariate_analysis(df):

    # start of univariate analysis
    import plotly.io as pio
    pio.renderers
    pio.renderers.default = 'browser'

    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.impute import SimpleImputer


    # Mostly affected country
    print("\nMostly affected country")
    fig = px.pie(df, names='Countries', template='seaborn')
    fig.update_traces(rotation=90, pull=[0.2,0.03,0.1,0.03,0.1], textinfo="percent+label", showlegend=False)
    # fig.show()
    save_plotly_image(fig,"mostaffectedcntry")

    # Mostly affected sector
    print("\nMostly affected sector")
    fig = px.pie(df, names='Industry Sector', template='seaborn')
    fig.update_traces(rotation=90, pull=[0.2,0.03,0.1,0.03,0.1], textinfo="percent+label", showlegend=False)
    # fig.show()
    save_plotly_image(fig,"mostaffectedsec")

    # Mostly affected local
    print("\nMostly affected local")
    fig = px.pie(df, names='Local', template='seaborn')
    fig.update_traces(rotation=90, pull=[0.2,0.03,0.1,0.03,0.1], textinfo="percent+label", showlegend=False)
    # fig.show()
    save_plotly_image(fig,"mostaffectedlocal")

    # Mostly affected gender
    print("\nMostly affected Employee type")
    fig = px.pie(df, names='Employee type', template='seaborn')
    fig.update_traces(rotation=90, pull=[0.2,0.03,0.1,0.03,0.1], textinfo="percent+label", showlegend=False)
    # fig.show()
    save_plotly_image(fig, "mostaffectedempltyp")

    # Mostly affected gender
    print("\nMostly affected gender")
    fig = px.pie(df, names='Gender', template='seaborn')
    fig.update_traces(rotation=90, pull=[0.2,0.03,0.1,0.03,0.1], textinfo="percent+label", showlegend=False)
    # fig.show()
    save_plotly_image(fig, "mostaffectedgender")

    # Mostly observed Risk factor
    print("\nMostly observed Risk factor")
    fig = px.pie(df, names='Critical Risk', template='seaborn')
    fig.update_traces(rotation=90, pull=[0.2,0.03,0.1,0.03,0.1], textinfo="percent+label", showlegend=False)
    save_plotly_image(fig,"mostobserverdriskfactor")

    # Mostly affected quarter
    print("\nMostly affected quarter")
    fig = px.pie(df, names='Quarter', template='seaborn')
    fig.update_traces(rotation=90, pull=[0.2,0.03,0.1,0.03,0.1], textinfo="percent+label", showlegend=False)
    # fig.show()
    save_plotly_image(fig, "mostaffectedquarter")

    # Mostly affected season
    print("\nMostly affected season")
    fig = px.pie(df, names='Season', template='seaborn')
    fig.update_traces(rotation=90, pull=[0.2,0.03,0.1,0.03,0.1], textinfo="percent+label", showlegend=False)
    # fig.show()
    save_plotly_image(fig, "mostaffectedseason")

    print("\nPotential Accident Level counts")
    #plt.figure(figsize=(20,20))
    sns.countplot(y='Potential Accident Level', data=df, label="Count", orient='h')
    # plt.show()
    save_image("unimul_potacclevel")

    print("\nAccident Level")
    #plt.figure(figsize=(20,20))
    sns.countplot(y='Accident Level', data=df, label="Count", orient='h')
    # plt.show()
    save_image("unimul_acclevel")

    print("\nCritical Risk counts")
    plt.figure(figsize=(20,20))
    sns.countplot(y='Critical Risk', data=df, label="Count", orient='h')
    # plt.show()
    save_image("unimul_critrisk")

    # # Accident Impact
    # print("\nAccident Impact")
    # df.plot(x='Date', y='Accident Impact', figsize=(15,4), kind='line')

    # end of univariate analysis

    # start of multivariate analysis

    # Analysis of Gender wrt Accident/ Potential Accident Level
    print("\n\nAnalysis of Gender wrt Accident/ Potential Accident Level")
    target_count(df, 'Gender')

    # Analysis of Employee type wrt Accident/ Potential Accident Level
    print("\n\nAnalysis of Employee type wrt Accident/ Potential Accident Level")
    target_count(df, 'Employee type')

    # Analysis of Industry Sector wrt Accident/ Potential Accident Level
    print("\n\nAnalysis of Industry Sector wrt Accident/ Potential Accident Level")
    target_count(df, 'Industry Sector')

    # Analysis of Country wrt Accident/ Potential Accident Level
    print("\n\nAnalysis of Country wrt Accident/ Potential Accident Level")
    target_count(df, 'Countries')

    # Analysis of Critical Risk wrt Accident/ Potential Accident Level
    print("\n\nAnalysis of Critical Risk wrt Accident/ Potential Accident Level")
    target_count(df, 'Critical Risk')

    # Analysis of Local wrt Accident/ Potential Accident Level
    print("\n\nAnalysis of Local wrt Accident/ Potential Accident Level")
    target_count(df, 'Local')

    # Analysis of Quarter wrt Accident/ Potential Accident Level
    print("\n\nAnalysis of Quarter wrt Accident/ Potential Accident Level")
    target_count(df, 'Quarter')

    # Analysis of Season wrt Accident/ Potential Accident Level
    print("\n\nAnalysis of Season wrt Accident/ Potential Accident Level")
    target_count(df, 'Season')

    # Analysis of Month wrt Accident/ Potential Accident Level
    print("\n\nAnalysis of Month wrt Accident/ Potential Accident Level")
    target_count(df, 'Month')

    # Analysis of Year wrt Accident/ Potential Accident Level
    print("\n\nAnalysis of Year wrt Accident/ Potential Accident Level")
    target_count(df, 'Year')

    # Analysis of Weekday wrt Accident/ Potential Accident Level
    print("\n\nAnalysis of Weekday wrt Accident/ Potential Accident Level")
    target_count(df, 'Weekday')

    # Analyze Country and Employee Type
    print("\n\nAnalyze Country and Employee Type")
    sns.countplot(x="Countries", data=df, hue="Employee type")
    # plt.show()
    save_image("unimulcntyempl")
    # Analyze Country and Industry Sector
    print("\n\nAnalyze Country and Industry Sector")
    sns.countplot(x="Countries", data=df, hue="Industry Sector")
    # plt.show()
    save_image("unimulcntyindsec")

    # Analyze Country and Gender
    print("\n\nAnalyze Country and Gender")
    sns.countplot(x="Countries", data=df, hue="Gender")
    # plt.show()
    save_image("unimulcntygend")

    # Analyze Employee Type and Gender
    print("\n\nAnalyze Employee Type and Gender")
    sns.countplot(x="Employee type", data=df, hue="Gender")
    # plt.show()
    save_image("unimulemptypgen")

    # Analyze Critical Risk and Gender
    print("\n\nAnalyze Critical Risk and Gender")
    plt.figure(figsize=(20, 25))
    sns.countplot(x="Critical Risk", data=df, hue="Gender", orient='h')
    # plt.show()
    save_image("unimulcritriskgen")

    # Analyze Local and Gender
    print("\n\nAnalyze Local and Gender")
    plt.figure(figsize=(15, 15))
    sns.countplot(x="Local", data=df, hue="Gender")
    # plt.show()
    save_image("unimullocalgen")
    # Analyze accident trend by Gender & Industry sector
    print("\n\nAnalyze accident trend by Gender & Industry sector")
    order = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5}
    fig = sns.FacetGrid(df, aspect=1.2, palette="winter", hue='Gender', col='Industry Sector', legend_out=True)
    fig.map(sns.countplot, 'Accident Level', order=order)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.show()
    save_image("unimulgendindsec")
    # Analyze potential accident trend by Gender & Industry sector
    print("\n\nAnalyze potential accident trend by Gender & Industry sector")
    fig = sns.FacetGrid(df, aspect=1.2, palette="winter", hue='Gender', col='Industry Sector', legend_out=True)
    fig.map(sns.countplot, 'Potential Accident Level', order=order)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.show()
    save_image("unimulgendindsec1")

    # Analyze accident trend year wise
    print("\n\nAnalyze accident trend year wise")
    acc_trend = df.pivot_table(index='Month', columns=['Year', 'Accident Level'], aggfunc='count')['Countries']
    acc_trend.replace(np.nan, 0, inplace=True)
    acc_trend[2016]

    df['Data'] = pd.to_datetime(df['Data'])
    df.groupby('Data').count()['Local'].plot(figsize=(15, 4))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    acc_trend[2016].plot(kind='bar', figsize=(15, 4), width=0.9, cmap='cool', title='2016')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    acc_trend[2017].plot(kind='bar', figsize=(15, 4), width=0.9, cmap='hot', title='2017')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.show()
    save_image("unimulacctrend")

    # accidents with the level I are most common. these are your small negligences, like when people forgot their PPE, or when they drop a tool, etc.
    # by looking at 2016 data, you can see that number of incidents in the first half of the year seem to be higher than in the second.
    # The overview for this dataset mentioned that this data is from manufacturing plants in South America, so the first half of the year is cold and the second is warmer.
    # So the number of incidents is higher in colder months.

    # # Analyze Accident Impact year wise
    # print("\n\nAnalyze Accident Impact year wise")
    # df.plot(x='Data', y='Accident Impact', figsize=(15, 4), kind='line')
    # plt.text(x='2016-6-15', y=28.5, s='July 2016', color='red', fontsize=12)
    # plt.vlines(x='2016-7-1', ymin=25, ymax=28, color='red', linestyles=':', linewidth=3)
    # plt.text(x='2017-3-5', y=28.5, s='March 2017', color='red', fontsize=12)
    # plt.vlines(x='2017-3-15', ymin=25, ymax=28, color='red', linestyles=':', linewidth=3)
    # plt.figure(figsize=(15, 10))
    # sns.boxplot(x='Month', y='Accident Impact', data=df, palette='Set3', saturation=1)
    # plt.show()

    # Analyze Accidents per Industry Sector WEEKDAY wise
    print("\n\nAnalyze Accidents per Industry Sector WEEKDAY wise")
    plt.figure(figsize=(15, 15))
    wd_order = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    week_d = df.pivot_table(index='Weekday', columns='Industry Sector', aggfunc='count')['Accident Level']
    week_d.loc[wd_order].plot(figsize=(10, 4), xticks=range(7), cmap='Dark2', kind='line')
    plt.ylabel('Number of accidents')
    # plt.show()
    save_image("unimulacctrendWD")

    # # Analyze Accident Impact per Industry Sector
    # print("\n\nAnalyze Accident Impact per Industry Sector")
    # df_ind = df.groupby('Industry Sector').count()['Data']
    # plt.figure(figsize=(15, 15))
    # df_ind.plot(kind='pie', figsize=(5, 5), cmap='Set2', autopct='%.2f', title='Number of Incidents')
    # plt.show()
    # df_ind_imp = df.groupby('Industry Sector')['Accident Impact'].mean()
    # df_ind_imp.plot(kind='pie', figsize=(5, 5), cmap='Set1', autopct='%.2f', title='Mean Accident Impact')
    # plt.show()

    # Analyze Accident Level vs Risk per Industry sector
    print("\n\nAnalyze Accident Level vs Risk per Industry sector")
    riskdf = df.copy()
    acc_cr = riskdf.pivot_table(index='Critical Risk', columns='Accident Level', aggfunc='count')['Month']
    acc_cr.replace(np.nan, 0, inplace=True)
    acc_cr['total'] = acc_cr.sum(axis=1)
    acc_cr.style.background_gradient(cmap='Blues')
    acc_cr.drop('Others', axis=0, inplace=True)

    acc_cr.nlargest(6, 'total').style.background_gradient(cmap='winter')
    acc_ind_risk = df.pivot_table(index='Critical Risk', columns='Industry Sector', aggfunc='count')['Accident Level']
    acc_ind_risk.replace(np.nan, 0, inplace=True)
    acc_ind_risk['total'] = acc_ind_risk.sum(axis=1)
    acc_ind_risk.nlargest(6, 'total').plot(kind='bar', xticks=range(30), figsize=(15, 5), cmap='summer')
    plt.xticks(rotation=90)

    acc_ind_risk_nt = acc_ind_risk.drop('total', axis=1)
    fig = acc_ind_risk_nt.plot(kind='bar', xticks=range(30), yticks=range(0, 21), figsize=(15, 6), cmap='Paired',
                               width=0.9)
    fig.set_facecolor('#2B2B2B')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.text(x=0.9, y=11, s='V', color='#CD661D', fontsize=25)
    plt.text(x=3.3, y=16, s='V', color='#ADD8E6', fontsize=25)
    plt.text(x=5.4, y=11, s='V', color='#ADD8E6', fontsize=25)
    plt.text(x=14.4, y=15, s='V', color='#ADD8E6', fontsize=25)
    plt.text(x=18.4, y=17.5, s='V', color='#ADD8E6', fontsize=25)
    plt.text(x=21.6, y=10, s='V', color='#FFD39B', fontsize=25)
    plt.text(x=28.6, y=9, s='V', color='#FFD39B', fontsize=25)
    plt.text(x=30, y=14, s='V', color='#CD661D', fontsize=25)

    return


# function to convert month variable into seasons
def month2seasons(x):
    if x in [9, 10, 11]:
        season = 'Spring'
    elif x in [12, 1, 2]:
        season = 'Summer'
    elif x in [3, 4, 5]:
        season = 'Autumn'
    elif x in [6, 7, 8]:
        season = 'Winter'
    return season
# function to convert month variable into quarters
def month2quarter(x):
    if x in [10, 11, 12]:
        quarter = 'Fourth'
    elif x in [1, 2, 3]:
        quarter = 'First'
    elif x in [4, 5, 6]:
        quarter = 'Second'
    elif x in [7, 8, 9]:
        quarter = 'Third'
    return quarter

# function to convert date to Year / Month / Day / Weekday/ WeekOfYear
def date2YMDW(df):
    df['Data'] = pd.to_datetime(df['Data'])
    df['Year'] = df['Data'].apply(lambda x : x.year)
    df['Month'] = df['Data'].apply(lambda x : x.month)
    df['Day'] = df['Data'].apply(lambda x : x.day)
    df['Weekday'] = df['Data'].apply(lambda x : x.day_name())
    df['WeekofYear'] = df['Data'].apply(lambda x : x.weekofyear)
    return df


# Helper function for relation between Accident Level/Potential Accident levels and other labels
def target_count(df, col1):
    orientation = 'v'
    fig = plt.figure(figsize=(15, 7.2))
    ax = fig.add_subplot(121)
    if col1 == 'Critical Risk':
        orientation = 'h'
    sns.countplot(x=col1, data=df, ax=ax, orient=orientation,
                  hue='Accident Level').set_title(col1.capitalize() + ' count plot by Accident Level', fontsize=13)
    plt.legend(labels=df['Accident Level'].unique())
    plt.xticks(rotation=90)

    ax = fig.add_subplot(122)
    sns.countplot(x=col1, data=df, ax=ax, orient=orientation,
                  hue='Potential Accident Level').set_title(
        col1.capitalize() + ' count plot by Potential Accident Level', fontsize=13)
    plt.legend(labels=df['Potential Accident Level'].unique())
    plt.xticks(rotation=90)
    save_image("targetcount" + col1)
    # return plt.show()
    return

def get_cv_scores(model, ftrain, ttarget):
    scores = cross_val_score(model, ftrain, ttarget, cv=10)
    cvscore = np.mean(scores)
    print('Cross Validation score: ', cvscore,'\n')
    return cvscore

####################################################################################################################################

def get_roc_auc_scores(model, ftrain, ttarget):
    y_pred_prob = logreg.predict_proba(ftrain)[:, 1]
    scores = roc_auc_score(ttarget, model.predict_proba(ftrain), multi_class='ovr')
    rocaucscore = np.mean(scores)
    print('roc auc score: ', rocaucscore,'\n')
    return rocaucscore

####################################################################################################################################
from sklearn.model_selection import RepeatedStratifiedKFold

def gridsearchResult(model, score, grid_values, ftrain, ttarget):
    if (score == ''):
      score = 'accuracy'
    # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    cv=5
    grid = GridSearchCV(estimator=model, param_grid=grid_values, scoring=score, verbose=0, n_jobs=-1, cv=cv)
    grid_result = grid.fit(ftrain, ttarget.values.ravel())
    print('\n')
    print('Best Score: ', grid_result.best_score_)
    print('Best Params: ', grid_result.best_params_)
    print('Best Estimator: ', grid_result.best_estimator_)
    print('\n')
    return grid_result

def make_tree(path):
    tree = dict(name=path, children=[])
    try: lst = os.listdir(path)
    except OSError:
        pass #ignore errors
    else:
        for name in lst:
            # fn = os.path.join(path, name)
            fn = name

            if os.path.isdir(fn):
                tree['children'].append(make_tree(fn))
            else:
                tree['children'].append(dict(name=fn))
    return tree

def training_DL_data_preparation():

    max_features = 5500
    maxlen = 200
    label  = 'Accident Level'

    preproc_path = 'C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\training\\'

    path = 'C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\training\\'

    all_files = glob.glob(preproc_path + "/prepro*.csv")

    print("list of files for DL training")

    for i in all_files:
        print(i)

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, engine='python', usecols=["Industry Sector", "Gender", "Employee type", "Critical Risk",
                                                           "Description_denoised", "Accident Level"], index_col=None, header=0)
        li.append(df)
    df = pd.concat(li, axis=0, ignore_index=True)
    df.drop_duplicates(inplace=True)

    # df = pd.read_csv(preproc_path + 'prepro_data.csv', usecols=["Industry Sector", "Gender", "Employee type", "Critical Risk",
    #                                                             "Description_denoised", "Accident Level"])

    Y = pd.get_dummies(df[label]).values
    x_train, x_test, y_train, y_test = train_test_split(df.Description_denoised, Y, test_size=0.15,
                                                        random_state=0)
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(x_train)
    tokenized_train = tokenizer.texts_to_sequences(x_train)
    x_train_seq = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
    tokenized_test = tokenizer.texts_to_sequences(x_test)
    x_test_seq = sequence.pad_sequences(tokenized_test, maxlen=maxlen)
    tokenizer.fit_on_texts(df['Description_denoised'].values)
    word_index = tokenizer.word_index

    filename = path + "glove.wordindex.json"
    with open(filename, 'w') as f:
        f.write(json.dumps(word_index))

    # session['wordindex'] = word_index
    print('Found %s unique tokens.' % len(word_index))

    df_lstm_train_array = np.c_[x_train_seq, y_train]
    df_lstm_test_array = np.c_[x_test_seq, y_test]

    save(path  + 'lstm_train_data.npy', df_lstm_train_array)
    save(path  + 'lstm_test_data.npy', df_lstm_test_array)


    return


def training_ML_data_preparation():

    label = 'Accident Level'

    reloaded_word_vectors = \
        KeyedVectors.load('C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\training\\vectors.kv')

    preproc_path = 'C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\training\\'

    path = 'C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\training\\'

    all_files = glob.glob(preproc_path + "/prepro*.csv")

    print("list of files for ML training")

    for i in all_files:
        print(i)

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, engine='python', usecols=["Industry Sector", "Gender", "Employee type", "Critical Risk",
                                                           "Description_denoised", "Accident Level"],index_col=None, header=0)
        li.append(df)
    df = pd.concat(li, axis=0, ignore_index=True)
    df.drop_duplicates(inplace=True)

    # df = pd.read_csv(preproc_path + 'prepro_data.csv', usecols=["Industry Sector", "Gender", "Employee type", "Critical Risk",
    #                                                             "Description_denoised", "Accident Level"])

    temp_dict = {
        "Mining": 0,
        "Metals": 1,
        "Others": 2,
    }

    df['Industry Sector'] = df['Industry Sector'].map(temp_dict).fillna(0).astype(int)


    # newdf_copy['Industry Sector'] = LabelEncoder().fit_transform(newdf_copy['Industry Sector']).astype(np.int8)

    temp_dict = {
        "Male": 0,
        "Female": 1,
    }
    df['Gender'] = df['Gender'].map(temp_dict).fillna(0).astype(int)
    # newdf_copy['Gender'] = LabelEncoder().fit_transform(newdf_copy['Gender']).astype(np.int8)

    temp_dict = {
        "Third Party": 0,
        "Employee": 1,
        "Third Party (Remote)": 2,
    }

    df['Employee type'] = df['Employee type'].map(temp_dict).fillna(0).astype(int)
    # newdf_copy['Employee type'] = LabelEncoder().fit_transform(newdf_copy['Employee type']).astype(np.int8)

    temp_dict = {
        "Pressed (Heavy)": 0,
        "Manual Tools": 1,
        "Pressed (Medium)": 2,
        "Pressed (Minor)": 3,
        "Chemical substances": 4,
        "Vehicles and Mobile equipment": 5,
        "Cut": 6,
        "Projection": 7,
        "Fall prevention (same level)": 8,
        "Venomous Animals": 9,
        "Fall": 10,
        "Bees": 11,
        "Foot Twist": 12,
        "Rock fall (Heavy)": 13,
        "Pressurized Systems": 14,
        "Projection of fragments": 15,
        "remains of choco": 16,
        "Fall prevention": 17,
        "Projection of mud": 18,
        "Rock fall (Medium)": 19,
        "Rock fall (Minor)": 20,
        "Suspended Loads": 21,
        "Fall (Medium)": 22,
        "Burn": 23,
        "Machine Part Fall": 24,
        "Liquid Metal": 25,
        "Fall (High)": 26,
        "Pressurized Systems / Chemical Substances": 27,
        "Power lock": 28,
        "Cut (Minor)": 29,
        "Blocking and isolation of energy": 30,
        "Machine Protection": 31,
        "Electrical Shock": 32,
        "Projection/Manual Tools": 33,
        "Plates": 34,
        "\nNot applicable": 35,
        "Pressed (Low)": 36,
        "Poll": 37,
        "Projection/Choco": 38,
        "Burn (Minor)": 39,
        "Cooking": 40,
        "Projection/Burning": 41,
        "Pressed": 42,
        "Traffic": 43,
        "Electrical shock": 44,
        "Fall (Low)": 45,
        "Others": 46,
        "Confined space": 47,
        "Electrical installation": 48,
        "Individual protectiion equipment": 49
    }
    df['Critical Risk'] = df['Critical Risk'].map(temp_dict).fillna(0).astype(int)

    df[label] = LabelEncoder().fit_transform(df[label]).astype(np.int8)

    outV = desc2vector(df['Description_denoised'].tolist(), df[label].tolist(),df["Industry Sector"].tolist(),df["Gender"].tolist(),
                   df["Employee type"].tolist(),df["Critical Risk"].tolist(), reloaded_word_vectors)

    ML_feature_df = pd.DataFrame(outV)

    ML_feature_df.to_csv(path + '\\' + 'ML_feature_df.csv')

    return

def desc2vector(desc, al, IS, gender, ET, CR, reloaded_word_vectors):
    mean_fvector = []
    mean_svector = []
    for nsent in range(len(desc)):
        # for isentence in desc[nsent].split():
        mean_svector = []
        input_list = desc[nsent].split()
        filtered_words = filter(lambda x: x in reloaded_word_vectors.vocab, input_list)
        sentence_len = 0.001
        vector = np.empty([200])
        for Sword in filtered_words:
            vector = np.add(vector, reloaded_word_vectors[Sword])
            sentence_len = sentence_len + 1
        mean_svector = np.divide(vector, sentence_len).tolist()
        mean_svector.extend([IS[nsent]])
        mean_svector.extend([gender[nsent]])
        mean_svector.extend([ET[nsent]])
        mean_svector.extend([CR[nsent]])
        mean_svector.extend([al[nsent]])
        mean_fvector.append(mean_svector)
    return mean_fvector

def train_bidirectional_lstm():
    print("Start of LSTM training")
    path = "C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\training\\"
    rep_filename = "model_perf_report.csv"
    rep_df = pd.read_csv(path + rep_filename, usecols=['Classifier Name', 'Test Accuracy'])

    embeddings_index = {}
    embed_size = 200
    RNN_units = 64
    embed_size = maxlen = 200
    epochs = 5
    batch_size = 64

    f = open(path + "glove.6B.200d.txt",encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    filename = path + "glove.wordindex.json"
    with open(filename) as f:
        word_index = json.loads(f.read())

    # word_index = session['wordindex']
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    embedding_matrix = np.random.normal(emb_mean, emb_std, (len(word_index) + 1, embed_size))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Defining Neural Network
    model = Sequential()
    # trainable embeddidng layer
    model.add(Embedding(len(word_index) + 1, output_dim=embed_size, weights=[embedding_matrix], input_length=maxlen))
    # LSTM
    model.add(LSTM(RNN_units, return_sequences=True))
    # Adding a dropout layer
    model.add(Dropout(0.2))
    model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.0001),
                   recurrent_regularizer=l2(0.0001), bias_regularizer=l2(0.0001)))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='softmax'))
    # compile the model
    adam = optimizers.Adam(lr=0.01, decay=1e-6)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    lstm_train_data = load(path + 'lstm_train_data.npy')
    lstm_test_data = load(path + 'lstm_test_data.npy')

    x_train_seq = lstm_train_data[:,0:200]
    y_train = lstm_train_data[:, 200:]

    x_test_seq = lstm_test_data[:, 0:200]
    y_test = lstm_test_data[:, 200:]
    history = model.fit(x_train_seq, y_train, batch_size=batch_size, validation_data=(x_test_seq, y_test),
                        # callbacks=[EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001)],
                        shuffle=False, epochs=epochs, use_multiprocessing=True)

    scores = model.evaluate(x_test_seq, y_test, verbose=0)

    pred = model.predict(x_test_seq)
    pred = (pred > 0.5)


    print('\nClassification Report\n{}'.format(classification_report(np.argmax(y_test,axis=1), np.argmax(pred,axis=1))))


    plot_confusion_matrix(confusion_matrix(np.argmax(y_test,axis=1), np.argmax(pred,axis=1)), classes=['0', '1', '2', '3', '4'],
                          normalize=False, title='LSTM Confusion matrix')

    LSTM_scores = ["LSTM", round(scores[1], 2) * 100]
    LSTM_scores

    if (rep_df[rep_df["Classifier Name"] == 'LSTM']["Test Accuracy"].values[0]) < round(scores[1] * 100, 2):
        lstm_path = "C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\trained_models\\"
        model.save(lstm_path + 'lstm.hdf5')
        rep_df.loc[rep_df['Classifier Name'] == 'LSTM', 'Test Accuracy'] = round(scores[1], 2) * 100
        rep_df.to_csv(path + rep_filename, encoding='utf-8')
        print("LSTM model saved on disk")

    print('Test accuracy LSTM:', LSTM_scores)
    print ("end  of lstm training")
    return

def train_svm_classifier():
    print("Start of SVM training")
    path = "C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\training\\"
    rep_filename = "model_perf_report.csv"
    rep_df = pd.read_csv(path + rep_filename, usecols=['Classifier Name', 'Test Accuracy'])

    ml_feature_df = pd.read_csv(path + 'ML_feature_df.csv')

    ml_feature_df.drop(['Unnamed: 0'], axis=1,inplace=True)
    ml_feature_df.drop_duplicates(inplace=True)
    y_temp = ml_feature_df.iloc[:, [-1]]
    x_temp = ml_feature_df.iloc[:, 0:204]
    X_trainml, X_validml, y_trainml, y_validml = train_test_split(x_temp, y_temp, test_size=0.2, random_state=0)

    svc = SVC(kernel='rbf', gamma=1, C=10, probability=True)
    # svc = SVC(kernel='rbf')

    svc.fit(X_trainml, y_trainml)

    print('SVM Classifier Scores after Scaling\n\n')
    print('SVM accuracy for train set: {0:.3f}'.format(svc.score(X_trainml, y_trainml)))
    print('SVM accuracy for test set: {0:.3f}'.format(svc.score(X_validml, y_validml)))
    test_score = svc.score(X_validml, y_validml)

    y_pred = svc.predict(X_validml)

    # # Classification Report
    # print('\n{}'.format(classification_report(y_validml, y_pred)))

    # Check cross validation score
    scores = cross_val_score(svc, y_validml, y_pred, cv=5)
    svc_cv_score = np.mean(scores)
    print('Cross Validation score: ', svc_cv_score, '\n')

    # Confusion Matrix
    # Finetuning of the model for identifying best parameters and accuracy scores
    grid_values = {'C': [0.1, 10],
                  'gamma': [1, 0.1],
                  'kernel': ['rbf']}


    svc = SVC(probability=True)
    grid_result = gridsearchResult(svc, 'accuracy', grid_values, X_trainml, y_trainml)
    svc_accuracy_score = grid_result.best_score_
    bestModel = grid_result.best_estimator_
    svc_best = bestModel.fit(X_trainml, y_trainml)
    pred = svc_best.predict(X_validml)

    # RMSE Score
    svc_rmse = np.sqrt(mean_squared_error(y_validml, pred))

    # ROC_AUC score
    # store the predicted probabilities
    y_pred_prob = svc_best.predict_proba(X_validml)[:, 1]
    y_pred_class = binarize([y_pred_prob], 0.3)[0]  # deciding the class of the 1st 10 records based on new threshold

    scores = roc_auc_score(y_validml, svc_best.predict_proba(X_validml), multi_class='ovr')
    svc_roc_auc_score = np.mean(scores)

    svc_test_score = svc_best.score(X_validml, y_validml)
    print("\nTest score: ", svc_test_score, " Roc Auc Score: ", svc_roc_auc_score, "Cross Validation score: ",
          svc_cv_score, " Best accuracy score: ", svc_accuracy_score, " Best RMSE score: ", svc_rmse, '\n')
    print("\nBest Model Training set Accuracy Score : ", svc_best.score(X_trainml, y_trainml))
    print("\nBest Model Test set Accuracy Score : ", svc_test_score)

    print('\nClassification Report\n{}'.format(classification_report(y_validml, pred)))

    plot_confusion_matrix(confusion_matrix(y_validml, pred), classes=['0', '1', '2', '3', '4'], normalize=False,
                          title=' SVM Confusion matrix')


    if (rep_df[rep_df["Classifier Name"] == 'SVM']["Test Accuracy"].values[0]) < round( svc_test_score * 100, 2):
        os.chdir('C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\trained_models\\')
        filename = 'svm.model'
        pickle.dump(svc_best, open(filename, 'wb'))
        rep_df.loc[rep_df['Classifier Name'] == 'SVM', 'Test Accuracy'] = round(svc_test_score * 100, 2)
        rep_df.to_csv(path + rep_filename, encoding='utf-8')
        print ("SVM Accurancy before save", round(svc_test_score * 100, 2) )
        print("SVM model saved on disk")

    svc_test_score = ["SVM", round(svc_test_score * 100, 2)]
    print('Test accuracy SVM:',svc_test_score)
    print ("end  of svm training")
    return

def train_ann_classifier():

    print("Start  of ANN training")
    global path, rep_filename
    path = "C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\training\\"
    rep_filename = "model_perf_report.csv"
    # rep_df = pd.read_csv(path + rep_filename, usecols=['Classifier Name', 'Test Accuracy'])
    ml_feature_df = pd.read_csv(path + 'ML_feature_df.csv')

    ml_feature_df.drop(['Unnamed: 0'], axis=1,inplace=True)
    ml_feature_df.drop_duplicates(inplace=True)
    y_temp = ml_feature_df.iloc[:, [-1]]
    y_temp = pd.get_dummies(y_temp.values.ravel())
    x_temp = ml_feature_df.iloc[:, 0:204]
    x_desc = ml_feature_df.iloc[:, 0:200]
    x_feat = ml_feature_df.iloc[:, 200:204]
    x_desc = x_desc/x_desc.max().max()
    x_temp = np.c_[x_desc, x_feat]

    X_trainml, X_validml, y_trainml, y_validml = train_test_split(x_temp, y_temp, test_size=0.2, random_state=0)

    def train_and_test_dropoutsgd(X_trainml, X_validml, y_trainml, y_validml, iterations, lr, hn, Lambda, bs, verb=True):
        rep_df = pd.read_csv(path + rep_filename, usecols=['Classifier Name', 'Test Accuracy'])
        global batchn_test_score, batch_cv_score, batchn_roc_auc_score, batchn_rmse
        iterations = iterations
        learning_rate = lr
        hidden_nodes = hn
        output_nodes = 5
        init_mode = 'he_uniform'
        dropout_rate = 0.2

        model = Sequential()
        model.add(Dense(hidden_nodes, input_shape=(204,), activation='relu',kernel_initializer=init_mode))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(hidden_nodes, activation='relu',kernel_initializer=init_mode))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_nodes, activation='softmax', kernel_regularizer=regularizers.l2(Lambda)))

        sgd = tf.keras.optimizers.Adam(lr=learning_rate, amsgrad=True)
        # Compile model
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        # Fit the model
        model.fit(X_trainml, y_trainml, epochs=iterations, batch_size=bs, verbose=0, validation_data=(X_validml, y_validml))
        test_loss, test_acc = model.evaluate(X_validml, y_validml, batch_size=bs, verbose=0)

        y_pred_class = model.predict(X_validml)
        # batchn_test_score = scores[1]

        print("\nAccuracy of the model on Training Data is - ", model.evaluate(X_trainml, y_trainml)[1] * 100)
        print("\nAccuracy of the model on Testing Data is - ", model.evaluate(X_validml, y_validml)[1] * 100)
        print('########################################################## ')
        print('Test accuracy is: {:0.4f} \nTest loss is: {:0.4f}'.
              format(test_acc, test_loss))
        print('########################################################## ')

        print('\nClassification Report\n{}'.format(classification_report(np.argmax(y_validml.values,axis=1), np.argmax(y_pred_class,axis=1))))

        # Confusion Matrix
        plot_confusion_matrix(confusion_matrix(np.argmax(y_validml.values,axis=1), np.argmax(y_pred_class,axis=1)), classes=['0', '1', '2', '3', '4'],
                              normalize=False, title=' ANN Confusion matrix')

        if (rep_df[rep_df["Classifier Name"] == 'ANN']["Test Accuracy"].values[0]) < round(test_acc * 100,2):
           ann_path = "C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\trained_models\\"
           model.save(ann_path + 'ann.hdf5')
           rep_df.loc[rep_df['Classifier Name'] == 'ANN', 'Test Accuracy'] = round(test_acc * 100, 2)
           rep_df.to_csv(path + rep_filename, encoding='utf-8')
           print("ANN model saved on disk")

        return model

    for k in range(1, 5):
        lr = math.pow(10, np.random.uniform(-7.0, 3.0))
        Lambda = math.pow(10, np.random.uniform(-7, -2))
        best_acc = train_and_test_dropoutsgd(X_trainml, X_validml, y_trainml, y_validml, 250, lr, 400, Lambda, 64)
        print("Try {0}/{1}: Best_val_acc: {2}, lr: {3}, Lambda: {4}\n".format(k, 100, best_acc, lr, Lambda))

    print("end  of ANN training")

def train_XGBOOST_classifier():
    print("Start of XGBoost training")
    path = "C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\training\\"
    rep_filename = "model_perf_report.csv"
    rep_df = pd.read_csv(path + rep_filename, usecols=['Classifier Name', 'Test Accuracy'])

    ml_feature_df = pd.read_csv(path + 'ML_feature_df.csv')
    ml_feature_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    ml_feature_df.drop_duplicates(inplace=True)
    y_temp = ml_feature_df.iloc[:, [-1]]
    x_temp = ml_feature_df.iloc[:, 0:204]
    X_trainml, X_validml, y_trainml, y_validml = train_test_split(x_temp, y_temp, test_size=0.2, random_state=0)

    xgb_cfl = XGBClassifier(learning_rate=0.1, n_estimators=500)

    xgb_cfl.fit(X_trainml, y_trainml)

    # Check cross validation score
    xgb_cv_score = get_cv_scores(xgb_cfl, X_trainml, y_trainml)

    y_pred = xgb_cfl.predict(X_validml)


    # RMSE Score
    xgb_rmse = np.sqrt(mean_squared_error(y_validml, y_pred))

    print('XGBoost Classifier\n\n')
    print('XGboost accuracy for train set: {0:.3f}'.format(xgb_cfl.score(X_trainml, y_trainml)))
    print('XGboost accuracy for test set: {0:.3f}'.format(xgb_cfl.score(X_validml, y_validml)))

    test_score = xgb_cfl.score(X_validml, y_validml)


    # RMSE Score
    xgb_rmse = np.sqrt(mean_squared_error(y_validml, y_pred))

    # ROC_AUC score
    # store the predicted probabilities
    y_pred_prob = xgb_cfl.predict_proba(X_validml)[:, 1]
    y_pred_class = binarize([y_pred_prob], 0.3)[0]  # deciding the class of the 1st 10 records based on new threshold

    scores = roc_auc_score(y_validml, xgb_cfl.predict_proba(X_validml), multi_class='ovr')
    xgb_roc_auc_score = np.mean(scores)

    print("\nTest score: ", test_score, " Roc Auc Score: ", xgb_roc_auc_score, "Cross Validation score: ",
          xgb_cv_score, " Best RMSE score: ", xgb_rmse, '\n')

    # Classification Report
    print('\n{}'.format(classification_report(y_validml, y_pred)))

    # Confusion Matrix
    plot_confusion_matrix(confusion_matrix(y_validml, y_pred), classes=['0', '1', '2', '3', '4'], normalize=False,
                          title='XGBoost Confusion matrix')

    if (rep_df[rep_df["Classifier Name"] == 'XGB']["Test Accuracy"].values[0]) < round(test_score * 100, 2):
        os.chdir('C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\trained_models\\')
        filename = 'xgb.model'
        pickle.dump(xgb_cfl, open(filename, 'wb'))
        rep_df.loc[rep_df['Classifier Name'] == 'XGB', 'Test Accuracy'] = round(test_score * 100, 2)
        rep_df.to_csv(path + rep_filename, encoding='utf-8')
        print("XGB model saved on disk")

    xgb_scores = ["XGB", round(test_score * 100, 2)]
    print('Test accuracy XGB:',xgb_scores)
    print ("end  of XGB training")
    return

def train_randomforest_classifier():
    print("Start of Random Forest training")
    path = "C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\training\\"
    rep_filename = "model_perf_report.csv"
    rep_df = pd.read_csv(path + rep_filename, usecols=['Classifier Name', 'Test Accuracy'])

    ml_feature_df = pd.read_csv(path + 'ML_feature_df.csv')
    ml_feature_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    ml_feature_df.drop_duplicates(inplace=True)
    y_temp = ml_feature_df.iloc[:, [-1]]
    x_temp = ml_feature_df.iloc[:, 0:204]
    X_trainml, X_validml, y_trainml, y_validml = train_test_split(x_temp, y_temp, test_size=0.2, random_state=0)

    scaler = MinMaxScaler()

    rf_cfl = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
    X_trainml_scaled = scaler.fit_transform(X_trainml)
    X_validml_scaled = scaler.transform(X_validml)

    rf_cfl.fit(X_trainml_scaled, y_trainml.values.ravel())

    print('Random Forest Classifier\n\n')
    print('Random Forest score for train set: {0:.3f}'.format(rf_cfl.score(X_trainml_scaled, y_trainml)))
    print('Random Forest score for test set: {0:.3f}'.format(rf_cfl.score(X_validml_scaled, y_validml)))

    test_score = rf_cfl.score(X_validml_scaled, y_validml)


    # Check cross validation score
    rfr_cv_score = get_cv_scores(rf_cfl, X_trainml_scaled, y_trainml)

    # Create the parameter grid based on the results of random search
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }

    rf_cfl = sklearn.ensemble.RandomForestClassifier()
    grid_result = gridsearchResult(rf_cfl, 'accuracy', param_grid, X_trainml_scaled, y_trainml)

    rfr_accuracy_score = grid_result.best_score_
    bestModel = grid_result.best_estimator_
    rfr_best = bestModel.fit(X_trainml_scaled, y_trainml)
    pred = rfr_best.predict(X_validml_scaled)

    rfr_rmse = np.sqrt(mean_squared_error(y_validml, pred))

    # ROC_AUC score
    # store the predicted probabilities
    y_pred_prob = rfr_best.predict_proba(X_validml_scaled)[:, 1]
    y_pred_class = binarize([y_pred_prob], 0.3)[0]  # deciding the class of the 1st 10 records based on new threshold

    scores = roc_auc_score(y_validml, rfr_best.predict_proba(X_validml_scaled), multi_class='ovr')
    rfr_roc_auc_score = np.mean(scores)

    rfr_test_score = rfr_best.score(X_validml_scaled, y_validml)
    print("\nTest score: ", rfr_test_score, " Roc Auc Score: ", rfr_roc_auc_score, "Cross Validation score: ",
          rfr_cv_score, " Best accuracy score: ", rfr_accuracy_score, " Best RMSE score: ", rfr_rmse, '\n')
    print("\nBest Model Training set Accuracy Score : ", rfr_best.score(X_trainml_scaled, y_trainml))
    print("\nBest Model Test set Accuracy Score : ", rfr_test_score)
    print('\nClassification Report\n{}'.format(classification_report(y_validml, pred)))
    # Confusion Matrix
    plot_confusion_matrix(confusion_matrix(y_validml, y_pred_class), classes=['0', '1', '2', '3', '4'],
                          normalize=False, title=' RF Confusion matrix')


    if (rep_df[rep_df["Classifier Name"] == 'RF']["Test Accuracy"].values[0]) < round(rfr_test_score * 100, 2):
        os.chdir('C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\trained_models\\')
        filename = 'rf.model'
        pickle.dump(rfr_best, open(filename, 'wb'))
        rep_df.loc[rep_df['Classifier Name'] == 'RF', 'Test Accuracy'] = round(rfr_test_score * 100, 2)
        rep_df.to_csv(path + rep_filename, encoding='utf-8')
        print("RandomForest model saved on disk")

    RF_scores = ["RF", round(rfr_test_score * 100, 2)]
    print('Test accuracy RF:',RF_scores)
    print ("end  of RF training")
    return


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()  # ta-da!

    return

def read_usercsv():
    data_path = "C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\Chatbot user details.csv"
    user_df = pd.read_csv(data_path)
    return user_df

def authenticate_user(user_df,username,password):
    is_authenticate = False
    userrole = ""
    auth_cond = (user_df['UserId'] == username) & (user_df['Password'] == password)
    is_authenticated = auth_cond.any()

    if is_authenticated:
        user_role = user_df[(user_df['UserId'] == username) & (user_df['Password'] == password)]['Role'].values
        userrole = user_role[0]
    return is_authenticated, userrole


def read_inputfiles(flask, path):

    if flask == 'Y':
        # data_path = "/content/drive/MyDrive/Colab Notebooks/project/Capstone/preprocessing/IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv"
        data_path = 'C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\preprocessing'

        all_files = glob.glob(data_path + "/*.csv")

        print("list of files for preprocessing")

        for i in all_files:
            print(i)

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, engine='python', index_col=None, header=0)
            li.append(df)
        df = pd.concat(li, axis=0, ignore_index=True)

    # if flask == 'P':
    #     data_path = "/content/drive/MyDrive/Colab Notebooks/project/Capstone/IHMStefanini_industrial_safety_and_health_database_with_accidents_description_EDA.csv"
    #     data_path = path
    #     df = pd.read_csv(data_path)
    #
    # if flask == 'N':
    #     data_path = "/content/drive/MyDrive/Colab Notebooks/project/Capstone/preprocessing/IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv"
    #     df = pd.read_csv(data_path)

    df.drop_duplicates(inplace=True)
    df = df.drop(['Unnamed: 0'], axis=1)
    df = shuffle(df)
    print("\nnumber of samples:", df.shape[0])
    df.head()
    # df_copy = df.loc[:, ['Accident Level', 'Potential Accident Level']]

    rep_path = "C:\\Users\\91992\\PycharmProjects\\CapstoneProject\\training\\model_perf_report.csv"
    rep_df = pd.read_csv(rep_path, usecols=['Classifier Name', 'Test Accuracy'])

    return df, rep_df


def _data_cleansing(df):
    import nltk
    nltk.download('stopwords')
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)

    def strip_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    # Removing the square brackets
    def remove_between_square_brackets(text):
        return re.sub('\[[^]]*\]', '', text)

    # Removing punctuations
    def remove_punctuations(text):
        return re.sub('[^a-zA-z0-9\s]', '', text)

    # CONVERT TO LOWERCASE
    def lowerCase(text):
        text = text.lower()
        return text

    # Removing the stopwords from text
    def remove_stopwords(text):
        final_text = []
        for i in text.split():
            if i.strip().lower() not in stop:
                final_text.append(i.strip())
        return " ".join(final_text)

    # Removing the noisy text
    def denoise_text(text):
        text = strip_html(text)
        text = remove_between_square_brackets(text)
        text = remove_punctuations(text)
        text = lowerCase(text)
        # text = remove_stopwords(text)
        return text

    # remove top 10 most common words assuming they will not help in additional information

    freq = pd.Series(' '.join(df['Description']).split()).value_counts()[:10]
    freq = list(freq.index)
    df['Description'] = df['Description'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    # remove top 25 least common words assuming they will not help in additional information

    freq = pd.Series(' '.join(df['Description']).split()).value_counts()[-25:]
    freq = list(freq.index)
    df['Description'] = df['Description'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    # Spell correct using Textblob
    df['Description'][:5].apply(lambda x: str(TextBlob(x).correct()))

    # Lemmatize the sentences

    df['Description'] = df['Description'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    df['Description_denoised'] = df['Description'].apply(denoise_text)

    df.reset_index(inplace=True)


    return df


def create_uniquetokens(Idf):
    tokens = Counter(re.findall(r'\w+', " ".join(Idf['Description_denoised'][:])))

    def F(word):
        return tokens.get(word)

    TOKEN_C = len(tokens)

    def P_c(word):
        return float(F(word)) / TOKEN_C

    # creating inverted index
    def create_index(data):
        index = defaultdict(list)
        for i, document in enumerate(data):
            for token in document.strip().split():
                index[token].append(i)
        return index

    inv_index = create_index(Idf['Description_denoised'][:])

    # sample analysis
    def P_x(word):
        sample = [Idf['Description_denoised'][:][i] for i in inv_index[word]]
        tokens_sample = Counter(re.findall(r'\w+', ' '.join(sample)))

        L_x = 0
        for k, v in tokens_sample.items():
            L_x += v
        return float(tokens_sample[word]) / L_x

    def kl_div(word):
        p_x = P_x(word)
        p_c = P_c(word)
        return p_x * log(p_x / p_c, 2)

    # collection analysis if the dataset is not huge

    def get_word_kl_metric(word):
        return df_kl_div.loc[df_kl_div.term == word]

        # tokens = Counter(re.findall(r'\w+', " ".join(Idf['Description_denoised'][:])))

    # inv_index = create_index(Idf['Description_denoised'][:])
    terms = list(tokens.keys())
    kl_div_val = []
    for t in terms:
        kl_div_val.append(kl_div(t))

    df_kl_div = pd.DataFrame({'term': terms, 'kl_div': kl_div_val}).sort_values(by='kl_div',
                                                                                ascending=True).reset_index(drop=True)
    df_kl_div.kl_div = df_kl_div.kl_div.astype(float)
    threshold = 0.015
    query = (df_kl_div['kl_div'] > threshold)
    stop_words = df_kl_div[query]['term']
    _not_stop_words = list(stop_words)

    return _not_stop_words


def filter_stopwords(df, _unique_words):
    # remove words that are in NLTK stopwords list
    nltk.download('stopwords')
    stop = set(stopwords.words('english'))

    new_stopwords = ["cm", "kg", "mr", "wa", "nv", "ore", "da", "pm", "am", "cx"]
    new_stopwords_list = stop.union(new_stopwords)

    final_stop_words = set([word for word in new_stopwords_list if word not in _unique_words])

    def remove_stopwords(text):
        final_text = []
        for i in text.split():
            if i.strip().lower() not in final_stop_words:
                final_text.append(i.strip())
        return " ".join(final_text)

    df['Description_denoised'] = df['Description_denoised'].apply(remove_stopwords)

    return df

if __name__ == "__main__":
    app.run(debug=True)
