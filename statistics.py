import pandas as pd
import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
    
import matplotlib.pyplot as plt

data_files = ["all_extracted.csv"]
df = pd.read_csv("all_extracted_data.csv", parse_dates=['publish_date'])

# df1 = pd.read_csv('all_extracted.csv')
# df2 = pd.read_csv()
# df3 = pd.read_csv()

#if Kashmir or Pak is in the headline
#label the headline accordindly
def label_place (row):
	kash = "Kashmir" in str(row['headline_text'])
	pak = "Pak" in str(row['headline_text'])
	if kash and pak:
		return "KP"
	elif kash and not pak:
		return "K"
	elif pak and not kash:
		return "P"

df['place'] = df.apply(lambda row: label_place(row), axis=1)

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

#show the statistics summary
print(rp.summary_cont(df['total_score']))
rp.summary_cont(df['total_score'].groupby(df['place'])).to_csv('statistics.txt', header=True, index=True, sep='\t', mode='a')

#ANOVA
st = open("statistics.txt", "a")
cor = stats.f_oneway(df['total_score'][df['place'] == 'KP'], 
             df['total_score'][df['place'] == 'K'],
             df['total_score'][df['place'] == 'P'])
st.write("Correlation: %d\npvalue: %d" %cor)

#post hoc tests

