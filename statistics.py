#statistics
import pandas as pd
import numpy as np
import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols

#sentiment analysis
#emotion
from wna.wnaffect import WNAffect
from wna.emotion import Emotion
wna = WNAffect('wn_corpus/wordnet-1.6/', 'wn_corpus/wn-domains-3.2/')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
#sid = SentimentIntensityAnalyzer()


cd = [(pd.Timestamp(2001,12,13), pd.Timestamp(2002,12,13)), (pd.Timestamp(2008,11,26), pd.Timestamp(2009,11,26)), (pd.Timestamp(2016,7,16),pd.Timestamp(2017,7,16))]
df = pd.read_csv("all_extracted.csv", parse_dates=['publish_date'])

def add_columns(date, text):
	conflict = label_conflict(date)
	emotion = label_emotion(text)
	return [conflict, emotion]

def label_conflict (date):
	if ((date >= cd[0][0]) & (date <= cd[0][1])):
		return "Standoff"
	elif ((date >= cd[1][0]) & (date <= cd[1][1])):
		return "Mumbai"
	elif ((date >= cd[2][0]) & (date <= cd[2][1])):
		return "Burhan"
	else:
		return "Non-conflict"

def label_emotion(text):
	for pair in pos_tag(word_tokenize(text)):
		emo = wna.get_emotion(pair[0], pair[1])
		if emo != None:
			if emo.level >= 5:
				if emo.get_level(5).name == "negative-fear":
					return "fear"
	return "non fear"


df[['conflict', 'emotion']] = df.apply(
    lambda row: pd.Series(add_columns(row['publish_date'],row['headline_text'])), axis=1)
df["total_score"] = df["pos_score"] + df["neg_score"]


'''
#HYPOTHESIS 1 Kashmir over Pakistan

#HYPOTHESIS 2 Kashmir conflict to Kashmir non conflict
print(rp.summary_cont(df['total_score'].groupby(df['conflict'])))
cor = stats.f_oneway(df['total_score'][(df['conflict'] == "Standoff") & (df['is_kashmir']==True)], 
             df['total_score'][(df['conflict'] == "Mumbai") & (df['is_kashmir']==True)],
             df['total_score'][(df['conflict'] == "Burhan") & (df['is_kashmir']==True)],
             df['total_score'][(df['conflict'] == "Non-conflict") & (df['is_kashmir']==True)])
print(cor) #st.write("Correlation: %d\npvalue: %d" % cor)


#HYPOTHESIS 3 Pakistan conflict to Pak non conflict
cor = stats.f_oneway(df['total_score'][(df['conflict'] == "Standoff") & (df['is_pakistan']==True)], 
             df['total_score'][(df['conflict'] == "Mumbai") & (df['is_pakistan']==True)],
             df['total_score'][(df['conflict'] == "Burhan") & (df['is_pakistan']==True)],
             df['total_score'][(df['conflict'] == "Non-conflict") & (df['is_pakistan']==True)])
print(cor) #st.write("Correlation: %d\npvalue: %d" % cor)


#HYPOTHESIS 4 Kashmir election to Kash non election
kash = df[df["is_kashmir"] == True]
print(kash.head())
print(kash.resample('Y', on="publish_date").mean())


#HYPOTHESIS 5 Pak election to Pak non election
pak = df[df["is_pakistan"] == True]
print(pak.head())
print(pak.resample('Y', on="publish_date").mean())

#HYPOTHESIS 6 Kash affect to Kash affect

'''

# df1 = pd.read_csv('all_extracted.csv')
# df2 = pd.read_csv()
# df3 = pd.read_csv()

#track the statistics
'''

st.write("\nExploratory Statistics for %s" % data_files[i])
st.write("\nTotal Entries: %s" % df['total_score'].count())
st.write("\nMean: %s" % df['total_score'].mean())
st.write("\nMedian: %s" % df['total_score'].median())
st.write("\nSum: %s" % df['total_score'].sum())
st.write("\nMax: %s" % df['total_score'].max())
st.write("\nMin: %s" % df['total_score'].min())
st.write("\nSTD: %s" % df['total_score'].std())
st.write("\nSE: %s" % df['total_score'].sem())
st.write("\nVariance: %s" % df['total_score'].var())
'''

'''
#show the statistics summary
print(rp.summary_cont(df['total_score']))
rp.summary_cont(df['total_score'].groupby(df['place'])).to_csv('statistics.txt', header=True, index=True, sep='\t', mode='a')

#ANOVA

cor = stats.f_oneway(df['total_score'][df['place'] == 'KP'], 
             df['total_score'][df['place'] == 'K'],
             df['total_score'][df['place'] == 'P'])
st.write("Correlation: %d\npvalue: %d" % cor)

#post hoc tests

'''