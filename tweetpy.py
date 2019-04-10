# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:39:50 2019

@author: MEDHA
"""
import tweepy #https://github.com/tweepy/tweepy
import csv
from tweepy.auth import OAuthHandler
import json
#Twitter API credentials
consumer_key = "xxxx"
consumer_secret = "yyyy"
access_key = "aaaa"
access_secret = "bbbb"


def get_all_tweets(screen_name):
    #Twitter only allows access to a users most recent 3240 tweets with this method

    #authorize twitter, initialize tweepy
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    #initialize a list to hold all the tweepy Tweets
    alltweets = []  

    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    #save most recent tweets
    alltweets.extend(new_tweets)

    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print ("getting tweets before %s" % (oldest))

        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)

        #save most recent tweets
        alltweets.extend(new_tweets)

        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        print ("%s tweets downloaded" % (len(alltweets)))
    #dumping all tweets into json
    status = alltweets[0]
    json.dumps(status._json)
    #getting the images in a tweet stored in variable a 
    i=0
    a=None
    if((True in [medium['type'] == 'photo' for medium in tweet.entities['media']])for tweet in alltweets):
        i=i+1
            
    else:
        a="None"
    a=i
    i=0
 #transform the tweepy tweets into a 2D array that will populate the csv 
    outtweets = [[tweet.id_str, tweet.created_at.strftime('%Y-%m-%d %H:%M:%S'), tweet.text.encode("utf-8"),tweet.favorite_count,tweet.retweet_count,a] for tweet in alltweets]
    

    #write the csv  
    with open('%s_tweets.csv' % screen_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["id","created_at","text","favourites/likes","retweets","No. of images"])
        writer.writerows(outtweets)

    pass
if __name__ == '__main__':
    #pass in the username of the account you want to download
    get_all_tweets("midasIIITD")
