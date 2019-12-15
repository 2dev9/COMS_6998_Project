import pandas as pd
from wna.wnaffect import WNAffect
from wna.emotion import Emotion
wna = WNAffect('wn_corpus/wordnet-1.6/', 'wn_corpus/wn-domains-3.2/')
from nltk.tokenize import word_tokenize
from nltk import pos_tag

def add_columns(date, text):
	conflict = label_conflict(date)
	emotion = label_emotion(text)
	election =label_election(date)
	year = label_year(date)
	return [conflict, emotion, election, year]

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
	return "non-fear"

def label_election(date):
	return (date.year == 2004) | (date.year == 2009) | (date.year == 2014)

def label_year(date):
	return date.year

cd = [(pd.Timestamp(2001,12,13), pd.Timestamp(2002,12,13)), (pd.Timestamp(2008,11,26), pd.Timestamp(2009,11,26)), (pd.Timestamp(2016,7,16),pd.Timestamp(2017,7,16))]
df = pd.read_csv("../data/all_extracted.csv", parse_dates=['publish_date'])

df[['conflict', 'emotion', 'election', 'year']] = df.apply(
    lambda row: pd.Series(add_columns(row['publish_date'],row['headline_text'])), axis=1)

df = df.drop(["pak_sent_binary","kash_senti_binary", "binary_score","pos_score","neg_score"], axis = 1)
df.to_csv("../data/all_annotated.csv", index=False)
