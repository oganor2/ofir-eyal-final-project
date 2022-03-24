import tweepy

bearer_token = 'AAAAAAAAAAAAAAAAAAAAAMPtXQEAAAAATtr5iLCYbFid7Q8vorCObQsmW%2Fc%3Dwj5oPVELZjCpbfU5aPPnam5rKNt3XnqJKHfukorHRuP6HRBhAR'
consumer_key='rpUOfDRx8JKXj7ODdF5fhXxjD'
consumer_secret='1cM0adWVgZuT15MZmGZqthh0pLdiwY8w1mft31YK9OVl5qKB9k'
access_token='1475517247561117698-HvzvQx5suJWfEJtDCjtQg2i56dd2ER'
access_token_secret='CQMfU0vuEvdw66btZXmbL7uEwciWoScdwQQXZ5ODsw7Uh'

client = tweepy.Client(bearer_token=bearer_token,consumer_key=consumer_key,consumer_secret=consumer_secret,
                          access_token=access_token,access_token_secret=access_token_secret)
tweets = client.get_users_tweets(id=1193881722628255744,max_results=20)
print(type(tweets.data[0].text))
print(tweets.data[0].text)
print(len(tweets.data))
# print(tweets[1])
[print(tweets.data[i].text) for i in range(len(tweets.data))]
# x = lambda i: print(tweets.data[i].text)
# l = range(0, 20)
# for n in l:
#     x(n)
# run the model with the tweets as input