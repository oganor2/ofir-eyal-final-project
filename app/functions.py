import tweepy
import pandas as pd
import streamlit as st
#from transformers import AutoModel, AutoTokenizer
#from nltk.tokenize import TweetTokenizer
#tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
from transformers.models.bertweet.tokenization_bertweet import TweetTokenizer

from predict import predict_one_user
from src.utils import tokenizer


def print_tweets(user):
    bearer_token = 'AAAAAAAAAAAAAAAAAAAAAMPtXQEAAAAATtr5iLCYbFid7Q8vorCObQsmW%2Fc%3Dwj5oPVELZjCpbfU5aPPnam5rKNt3XnqJKHfukorHRuP6HRBhAR'
    consumer_key='rpUOfDRx8JKXj7ODdF5fhXxjD'
    consumer_secret='1cM0adWVgZuT15MZmGZqthh0pLdiwY8w1mft31YK9OVl5qKB9k'
    access_token='1475517247561117698-HvzvQx5suJWfEJtDCjtQg2i56dd2ER'
    access_token_secret='CQMfU0vuEvdw66btZXmbL7uEwciWoScdwQQXZ5ODsw7Uh'
    client = tweepy.Client(bearer_token=bearer_token,consumer_key=consumer_key,consumer_secret=consumer_secret,
                          access_token=access_token,access_token_secret=access_token_secret)
    # tweets = client.get_users_tweets(id=1193881722628255744,max_results=20)
    tweets = client.get_users_tweets(id=user, max_results=20)
    return (tweets)

def create_df_tweets(id, tweets):
    tweets_list  = list(tweets.data)
    id_list = [id]*len(tweets_list)
    d = {'id': id_list , 'tweets': tweets_list }
    df = pd.DataFrame(data=d)
    #st.write(tweets_list)
    #st.write(id_list)
    return(df)

def tokenize_tweets(sentence,tweet_tokenizer,bert_tokenizer):
    sentence = tweet_tokenizer.tokenize(sentence)
    user = lambda x: '@USER' if x[0]=='@' else x
    http = lambda x: 'HTTPURL' if x[:4]=='http' else x
    demojize = lambda x: emoji.demojize(x)
    sentence = map(user,sentence)
    sentence = map(http,sentence)
    sentence = map(demojize, sentence)
    sentence = list(sentence)
    sentence = bert_tokenizer(" ".join(sentence),padding='max_length')
    return sentence

def df_tokenized(df_tweets):
    df_tweets['text_tokenized'] = df_tweets['tweets'].progress_apply(lambda x: tokenize_tweets(x,TweetTokenizer(),tokenizer))
    df_tweets = df_tweets.dropna()
    return (df_tweets)



def model(df_id_tokenized_tweets):
    #
    dic_output = {}
    #
    return (dic_output)



def convert_values(dict):
    #
    convert_dict = {}
    #
    return (convert_dict)




