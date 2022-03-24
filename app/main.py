
import streamlit as st
import tweepy
from transformers.models.bertweet.tokenization_bertweet import TweetTokenizer
from nltk.tokenize import TweetTokenizer
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
from predict import predict_one_user, DemographicsModel, ClassifierHead
import pandas as pd
import numpy as np
import torch
import transformers
import emoji
from PIL import Image

### Configuration:
max_result = 100
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAMPtXQEAAAAATtr5iLCYbFid7Q8vorCObQsmW%2Fc%3Dwj5oPVELZjCpbfU5aPPnam5rKNt3XnqJKHfukorHRuP6HRBhAR'
consumer_key = 'rpUOfDRx8JKXj7ODdF5fhXxjD'
consumer_secret = '1cM0adWVgZuT15MZmGZqthh0pLdiwY8w1mft31YK9OVl5qKB9k'
access_token = '1475517247561117698-HvzvQx5suJWfEJtDCjtQg2i56dd2ER'
access_token_secret = 'CQMfU0vuEvdw66btZXmbL7uEwciWoScdwQQXZ5ODsw7Uh'
client = tweepy.Client(bearer_token=bearer_token, consumer_key=consumer_key, consumer_secret=consumer_secret,
                       access_token=access_token, access_token_secret=access_token_secret)
data_tweets_for_csv = pd.DataFrame()
data_model_for_csv = pd.DataFrame()


##input : user id and enviroment
##output : return the X tweets of the user id
def print_tweets(user, env):
    if (env == 'Online'):
        tweets = client.get_users_tweets(id=user, max_results=max_result)  # , start_time = '2021-06-01T00:00:00Z')
        # tweets = client.get_users_tweets(id=user, max_results=max_result, end_time = '2021-06-01T00:00:00Z')
        if not (tweets.data):
            st.write("0 tweets")
            return -1
    return tweets


##input : user name
##output : return  user id
def get_user_id(screen_name):
    if not client.get_user(username=screen_name).data:  # if the user was not found
        return -1
    return client.get_user(username=screen_name).data.id


##input : sentence,tweet_tokenizer,bert_tokenizer
##output : return Processed sentance
def tokenize_tweets(sentence, tweet_tokenizer, bert_tokenizer):
    sentence = tweet_tokenizer.tokenize(sentence)
    user = lambda x: '@USER' if x[0] == '@' else x
    http = lambda x: 'HTTPURL' if x[:4] == 'http' else x
    demojize = lambda x: emoji.demojize(x)
    sentence = map(user, sentence)
    sentence = map(http, sentence)
    sentence = map(demojize, sentence)
    sentence = list(sentence)
    sentence = bert_tokenizer(" ".join(sentence), padding='max_length')
    return sentence


##input : df contain user id and tweets
##output : return Processed df
def df_tokenized(df_tweets):
    df_tweets['text_tokenized'] = df_tweets['tweets'].apply(lambda x: tokenize_tweets(x, TweetTokenizer(), tokenizer))
    df_tweets = df_tweets.dropna()
    return (df_tweets)


##input : user id
##output : return  user name
def get_user_name(user_id):
    name = (client.get_user(id=user_id).data.username)
    return (name)


### main page
# image = Image.open('logo.jpg')
# st.image(image,width=450)

# title = '<p style="color:Blue; font-size: 35px;">Final Project</p>'
# st.markdown(title, unsafe_allow_html=True)

# title2 = '<p style="color:Black; font-size: 27px;">Please enter a User Name</p>'
# st.markdown(title2, unsafe_allow_html=True)
env = st.selectbox("Choose Online/Demo", ['Online', 'Demo'])
# env = st.selectbox("", ['Online', 'Demo'])
screen_name = st.text_input('Please enter a User Name')

if env == 'Demo':
    if screen_name == '':
        st.write("Enter a user name")
    if screen_name.lower() == 'hillaryclinton':
        data_tweets_for_csv = pd.read_csv('C:/Users/oganor/PycharmProjects/DemoData/Tweets_data_hillaryclinton.csv')
        data_model_for_csv = pd.read_csv('C:/Users/oganor/PycharmProjects/DemoData/Model_data_hillaryclinton.csv')
    if screen_name.lower() == 'michelleobama':
        data_tweets_for_csv = pd.read_csv('C:/Users/oganor/PycharmProjects/DemoData/Tweets_data_MichelleObama.csv')
        data_model_for_csv = pd.read_csv('C:/Users/oganor/PycharmProjects/DemoData/Model_data_MichelleObama.csv')
    if screen_name.lower() == 'kendricklamar':
        data_tweets_for_csv = pd.read_csv('C:/Users/oganor/PycharmProjects/DemoData/Tweets_data_kendricklamar.csv')
        data_model_for_csv = pd.read_csv('C:/Users/oganor/PycharmProjects/DemoData/Model_data_kendricklamar.csv')
    if screen_name != '':
        data_model_for_csv = data_model_for_csv[["category", "attributes", "probability"]]
        st.table(data_model_for_csv)
        user_id = get_user_id(screen_name)
        with st.sidebar:
            csv_tweets = data_tweets_for_csv.to_csv()
            csv_model = data_model_for_csv.to_csv()
            st.write("User Name: ", screen_name)
            # st.write("User ID: ", user_id)
            st.download_button(
                "download tweets data", csv_tweets, "Tweets_data_" + screen_name + ".csv", "text/csv", key="download-csv")
            st.download_button(
                "download model data", csv_model, "Model_data_" + screen_name + ".csv", "text/csv", key="download-csv")


if env == 'Online':
    ### check if the username is legal
    username_check = screen_name.replace("_", "")
    if (screen_name != ''):
        if (not username_check.isalnum()):
            st.write('Error! Enter a correct user name')
            st.write('A username can only contain alphanumeric characters (letters A-Z, numbers 0-9) and underscores')
    # if screen_name != '':
        else:
            user_id = get_user_id(screen_name)
            if user_id == -1:
                st.write("User name was not found")
            else:
                st.balloons()
                st.write("The chosen User ID is", str(user_id))
                # wait = '<p style="color:#3f8098; font-size: 17px;">Searching for tweets.. please wait</p>'
                # st.markdown(wait, unsafe_allow_html=True)
                tweets = print_tweets(user_id, env)
                if tweets!=-1:
                    data = pd.DataFrame(columns=['tweets'])  # build df
                    for i in range(len(tweets.data)):
                        curr_tweet = tweets.data[i].text
                        # st.write(i + 1, curr_tweet)

                        ### remove Hebrew
                        res = ''.join([i for i in curr_tweet if (not i.isalnum()) or (i.isdigit()) or (
                                (i.isalpha() and i.isupper()) or (i.isalpha() and i.islower()))])

                        new_row = {'tweets': res}
                        data = data.append(new_row, ignore_index=True)
                    data['user_id'] = user_id
                    final_data = df_tokenized(data)

                    ### Add the model path
                    # model_path = 'C:/Users/oganor/PycharmProjects/models/BertFinetunedBinaryLabelsOnly'
                    # model_path = 'C:/Users/selalouf/code/SocialAI/BertFinetunedLessLabels'
                    # model_path = 'C:/Users/oganor/PycharmProjects/models/BertFinetunedBinaryLabelsOnly'
                    model_path = 'C:/Users/oganor/PycharmProjects/models/BertFinetunedLessLabels'
                    # model_path = 'C:/Users/oganor/PycharmProjects/models/BertFinetunedLessLabelsAugmented'
                    data_load_state = st.text('Loading tweets... please wait')
                    result_by_category = predict_one_user(final_data, model_path=model_path)
                    data_load_state.text('done!')
                    poli_curr_probs=result_by_category['political']
                    poli_curr_sum= poli_curr_probs[1] + poli_curr_probs[2]
                    poli_update_probs=[poli_curr_probs[1] / poli_curr_sum, poli_curr_probs[2] / poli_curr_sum]
                    #st.write(poli_update_probs)
                    result_by_category['political']=poli_update_probs

                    #st.write(result_by_category['race'])
                    race_curr_probs=result_by_category['race']
                    race_curr_sum= race_curr_probs[0] + race_curr_probs[2]
                    race_update_probs=[race_curr_probs[0] / race_curr_sum, race_curr_probs[2] / race_curr_sum]
                    print(race_update_probs)
                    result_by_category['race'] = race_update_probs

                    attributes = ('gender', 'political', 'children', 'race', 'income', 'education', 'age')
                    # attributes_dict =   ##binary
                    #                           {'gender': ['Male', 'Female'],
                    #                           'political': ['Republican', 'Democrat'],
                    #                           'children': ['Yes', 'No'],
                    #                           'race': ['Black', 'White'],
                    #                           'income': ['Over 35$', 'Under 35$'],
                    #                           'education': ['Degree', 'High School'],
                    #                           'age': ['Under 25', '25 and up']
                    #                    }
                    attributes_dict = {  ##less_lables
                        'gender': ['Female', 'Male'],
                        # 'political': ['Other', 'Republican', 'Democrat'],
                        'political': ['Republican', 'Democrat'],
                        'children': ['Yes', 'No'],
                        # 'race': ['White', 'Other', 'Black'],
                        'race': ['White', 'Black'],
                        'income': ['Under 35$', 'Over 35$'],
                        'education': ['High School', 'Degree'],
                        'age': ['25 and up', 'Under 25']
                    }


                    df = pd.DataFrame(columns=['category', 'attributes', 'probability'])

                    for cat, probs in result_by_category.items():
                        for i in range(len(probs)):
                            add_row = {'category': str(cat), 'attributes': attributes_dict[cat][i],
                                       'probability': probs[i]}
                            df = df.append(add_row, ignore_index=True)
                            # st.write(df)
                    df.groupby(['category'])
                    #st.write(df)
                    st.table(df)

                    data_tweets_for_csv = data[["user_id", "tweets"]].copy()
                    data_model_for_csv = df.copy()

            ###Create the side bar
        with st.sidebar:
            csv_tweets = data_tweets_for_csv.to_csv()
            csv_model = data_model_for_csv.to_csv()
            st.write("User Name: ", screen_name)
            # st.write("User ID: ", user_id)
            st.download_button(
                "download tweets data", csv_tweets, "Tweets_data_" + screen_name + ".csv", "text/csv", key="download-csv")
            st.download_button(
                "download model data", csv_model, "Model_data_" + screen_name + ".csv", "text/csv", key="download-csv")


