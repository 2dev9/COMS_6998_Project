#statistics
import pandas as pd
import numpy as np
import scipy.stats as stats
import researchpy as rp
import scikit_posthocs as sp

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
df = pd.read_csv("../data/all_extracted.csv", parse_dates=['publish_date'])

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

df[['conflict', 'emotion', 'election', 'year']] = df.apply(
    lambda row: pd.Series(add_columns(row['publish_date'],row['headline_text'])), axis=1)

#summary stats
summary = rp.summary_cont(df['total_score'].groupby(df['conflict']))
print(summary)

#HYPOTHESIS 1 Kashmir over Pakistan
print("HYPOTHESIS 1")
levene = stats.levene(df['total_score'][df['is_kashmir']==True], 
             df['total_score'][df['is_pakistan']==True])

print("Variance is equal %r" % (levene[1] >.05))
cor = stats.ttest_ind(df['total_score'][df['is_kashmir']==True], 
	df['total_score'][df['is_pakistan']==True], equal_var=(levene[1] >.05))
print(cor)

print(rp.summary_cont(df.groupby(['is_kashmir','year'])['total_score']))
cor = stats.kruskal(df['total_score'][(df['year'] == 2001) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2002) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2003) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2004) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2005) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2006) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2007) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2008) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2009) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2010) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2011) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2012) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2013) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2014) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2015) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2016) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2017) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2018) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2001) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2002) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2003) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2004) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2005) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2006) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2007) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2008) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2009) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2010) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2011) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2012) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2013) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2014) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2015) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2016) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2017) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2018) & (df['is_pakistan']==True)],)
print(cor)
#HYPOTHESIS 2 Kashmir conflict to Kashmir non conflict
print("HYPOTHESIS 2")
print(rp.summary_cont(df.groupby(['is_kashmir','conflict'])['total_score']))
levene = stats.levene(df['total_score'][(df['conflict'] == "Standoff") & (df['is_kashmir']==True)], 
             df['total_score'][(df['conflict'] == "Mumbai") & (df['is_kashmir']==True)],
             df['total_score'][(df['conflict'] == "Burhan") & (df['is_kashmir']==True)],
             df['total_score'][(df['conflict'] == "Non-conflict") & (df['is_kashmir']==True)])
print(levene)

if levene[1]>.05:
	print("ANOVA")
	cor = stats.f_oneway(df['total_score'][(df['conflict'] == "Standoff") & (df['is_kashmir']==True)], 
	             df['total_score'][(df['conflict'] == "Mumbai") & (df['is_kashmir']==True)],
	             df['total_score'][(df['conflict'] == "Burhan") & (df['is_kashmir']==True)],
	             df['total_score'][(df['conflict'] == "Non-conflict") & (df['is_kashmir']==True)])
else:
	cor = stats.kruskal(df['total_score'][(df['conflict'] == "Standoff") & (df['is_kashmir']==True)], 
	             df['total_score'][(df['conflict'] == "Mumbai") & (df['is_kashmir']==True)],
	             df['total_score'][(df['conflict'] == "Burhan") & (df['is_kashmir']==True)],
	             df['total_score'][(df['conflict'] == "Non-conflict") & (df['is_kashmir']==True)])
print(cor) #st.write("Correlation: %d\npvalue: %d" % cor)


#HYPOTHESIS 3 Pakistan conflict to Pak non conflict
print("HYPOTHESIS 3")
print(rp.summary_cont(df.groupby(['is_pakistan','year'])['total_score']))
levene = stats.levene(df['total_score'][(df['conflict'] == "Standoff") & (df['is_pakistan']==True)], 
             df['total_score'][(df['conflict'] == "Mumbai") & (df['is_pakistan']==True)],
             df['total_score'][(df['conflict'] == "Burhan") & (df['is_pakistan']==True)],
             df['total_score'][(df['conflict'] == "Non-conflict") & (df['is_pakistan']==True)])
print(levene)
if levene[1]>.05:
	cor = stats.f_oneway(df['total_score'][(df['conflict'] == "Standoff") & (df['is_pakistan']==True)], 
             df['total_score'][(df['conflict'] == "Mumbai") & (df['is_pakistan']==True)],
             df['total_score'][(df['conflict'] == "Burhan") & (df['is_pakistan']==True)],
             df['total_score'][(df['conflict'] == "Non-conflict") & (df['is_pakistan']==True)])
else:
	cor = stats.kruskal(df['total_score'][(df['conflict'] == "Standoff") & (df['is_pakistan']==True)], 
             df['total_score'][(df['conflict'] == "Mumbai") & (df['is_pakistan']==True)],
             df['total_score'][(df['conflict'] == "Burhan") & (df['is_pakistan']==True)],
             df['total_score'][(df['conflict'] == "Non-conflict") & (df['is_pakistan']==True)])
print(cor) #st.write("Correlation: %d\npvalue: %d" % cor)



#HYPOTHESIS 4 Kashmir election to Kash non election
print("HYPOTHESIS 4")

kedf = df['total_score'][(df['election']==True)& (df['is_kashmir']==True)]
knedf = df['total_score'][(df['election']==False) & (df['is_kashmir']==True)]
print("election kashmir")
print(rp.summary_cont(kedf.groupby(['year'])))
print("non election kashmir")
print(rp.summary_cont(kendf.groupby(['year'])))

levene = stats.levene(kedf, knedf)
print(levene)
cor = stats.ttest_ind(kedf, knedf, equal_var=(levene[1] >.05))

print(cor)
cor = stats.kruskal(df['total_score'][(df['year'] == 2001) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2002) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2003) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2004) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2005) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2006) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2007) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2008) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2009) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2010) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2011) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2012) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2013) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2014) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2015) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2016) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2017) & (df['is_kashmir']==True)],
	df['total_score'][(df['year'] == 2018) & (df['is_kashmir']==True)])
print(cor)
#print(sp.posthoc_conover(kedf.append(knedf), val_col='total_score', group_col='election'))

#HYPOTHESIS 5 Pak election to Pak non election
print("HYPOTHESIS 5")
pedf = df['total_score'][(df['election']==True)& (df['is_pakistan']==True)]
pnedf = df['total_score'][(df['election']==False) & (df['is_pakistan']==True)]
print("non election pakistan")
#print(rp.summary_cont(kendf.groupby(['year'])))
print("non election pakistan")
#print(rp.summary_cont(kendf.groupby(['year'])))
levene = stats.levene(pedf, pnedf)
print(levene)
cor = stats.ttest_ind(pedf, pnedf, equal_var=(levene[1] >.05))
print(cor)

cor = stats.kruskal(df['total_score'][(df['year'] == 2001) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2002) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2003) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2004) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2005) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2006) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2007) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2008) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2009) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2010) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2011) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2012) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2013) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2014) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2015) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2016) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2017) & (df['is_pakistan']==True)],
	df['total_score'][(df['year'] == 2018) & (df['is_pakistan']==True)])

print(cor)
#HYPOTHESIS 6 Kash affect to Kash affect in conflict
print("HYPOTHESIS 6")
kec = df[df["is_kashmir"]==True]
kcs = kec.groupby(['emotion', 'conflict'])
print("non election kashmir")
#print(rp.summary_cont(kcs))
print(stats.chisquare(kcs.size().divide(len(df)).multiply(100)))

#HYPOTHESIS 7 Pak affect to Pak affect in conflict
print("HYPOTHESIS 7")
pec = df[df["is_pakistan"]==True]
pcs = pec.groupby(['emotion', 'conflict'])
#print(rp.summary_cont(pcs))
print(stats.chisquare(pcs.size().divide(len(df)).multiply(100)))


