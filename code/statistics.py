#statistics
import pandas as pd
import scipy.stats as stats
import researchpy as rp
import scikit_posthocs as sp

import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp

df = pd.read_csv("../data/all_annotated.csv", parse_dates=['publish_date'])

tracker  = open('../outputs/stats.csv', "w", newline="")


#summary stats
tracker.write("Statistics\r\n")


#HYPOTHESIS 1 Kashmir over Pakistan
tracker.write("\nHYPOTHESIS 1: Kashmir-related headlines will have more negative sentiment scores on average than non-Kashmir-related  in any given year\r\n")
levene = stats.levene(df['total_score'][df['is_kashmir']==True], 
             df['total_score'][df['is_kashmir']==False])
tracker.write("Variance is equal: %r, %s\n" % ((levene[1] >.05), levene))

rp.summary_cont(df.groupby(['year'])['total_score']).to_csv(tracker, mode="a")
rp.summary_cont(df.groupby(['is_kashmir'])['total_score']).to_csv(tracker, mode="a")
tracker.write("Summary by year and relation to Kashmir\n")
rp.summary_cont(df.groupby(['is_kashmir','year'])['total_score']).to_csv(tracker, mode="a")
model = ols('total_score ~ C(year)*C(is_kashmir)', df).fit()
# Seeing if the overall model is significant
tracker.write("\nSeeing if overall model is significant\n")
tracker.write(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}\n")
tracker.write(str(model.summary()))

tracker.write("\nTwo-way ANOVA\n")
res = sm.stats.anova_lm(model, typ= 2)
res.to_csv(tracker, mode="a")

tracker.write("\nPost-hoc Tukey Tests\n")
mc = statsmodels.stats.multicomp.MultiComparison(df['total_score'], df['year'])
mc_results = mc.tukeyhsd()
tracker.write(str(mc_results))

mc = statsmodels.stats.multicomp.MultiComparison(df['total_score'], df['is_kashmir'])
mc_results = mc.tukeyhsd()
tracker.write(str(mc_results))

df['kashyear'] = df["is_kashmir"].astype(str) + df['year'].astype(str)
tracker.write("\nPost Hoc Student t-yTesting\n")
sp.posthoc_ttest(df, val_col='total_score', group_col='kashyear').to_csv(tracker, mode="a")


#HYPOTHESIS 2 Kashmir conflict to Kashmir non conflict
tracker.write("\n\n\n\nHYPOTHESIS 2: Kashmir-related headlines will have more negative sentiment scores on average in conflict periods than Kashmir-related headlines in non-conflict  periods\r\n")

rp.summary_cont(df.groupby(['is_kashmir','conflict'])['total_score']).to_csv(tracker, mode="a")

levene = stats.levene(df['total_score'][(df['conflict'] == "Standoff") & (df['is_kashmir']==True)], 
             df['total_score'][(df['conflict'] == "Mumbai") & (df['is_kashmir']==True)],
             df['total_score'][(df['conflict'] == "Burhan") & (df['is_kashmir']==True)],
             df['total_score'][(df['conflict'] == "Non-conflict") & (df['is_kashmir']==True)])
tracker.write("\nVariance is equal: %r, %s\n" % ((levene[1] >.05), levene))


if levene[1]>.05:
	cor = stats.f_oneway(df['total_score'][(df['conflict'] == "Standoff") & (df['is_kashmir']==True)], 
	             df['total_score'][(df['conflict'] == "Mumbai") & (df['is_kashmir']==True)],
	             df['total_score'][(df['conflict'] == "Burhan") & (df['is_kashmir']==True)],
	             df['total_score'][(df['conflict'] == "Non-conflict") & (df['is_kashmir']==True)])
	tracker.write("\nANOVA results: cor %f| p %f\n" % cor)
else:
	cor = stats.kruskal(df['total_score'][(df['conflict'] == "Standoff") & (df['is_kashmir']==True)], 
	             df['total_score'][(df['conflict'] == "Mumbai") & (df['is_kashmir']==True)],
	             df['total_score'][(df['conflict'] == "Burhan") & (df['is_kashmir']==True)],
	             df['total_score'][(df['conflict'] == "Non-conflict") & (df['is_kashmir']==True)])
	tracker.write("\nKruskal Wallis results: cor %f| p %f\n" % cor)

tracker.write("\nConover Post Hoc Test\n")
sp.posthoc_conover(df[df["is_kashmir"]==True], val_col='total_score', group_col='conflict').to_csv(tracker, mode="a")


#HYPOTHESIS 3 Pakistan conflict to Pak non conflict
tracker.write("\n\n\n\nHYPOTHESIS 3: Pakistan-related headlines will have more negative sentiment scores on average in conflict periods than Pakistan-related headlines in non-conflict  periods\r\n")

rp.summary_cont(df.groupby(['is_pakistan','conflict'])['total_score']).to_csv(tracker, mode="a")

levene = stats.levene(df['total_score'][(df['conflict'] == "Standoff") & (df['is_pakistan']==True)], 
             df['total_score'][(df['conflict'] == "Mumbai") & (df['is_pakistan']==True)],
             df['total_score'][(df['conflict'] == "Burhan") & (df['is_pakistan']==True)],
             df['total_score'][(df['conflict'] == "Non-conflict") & (df['is_pakistan']==True)])
tracker.write("\nVariance is equal: %r, %s\n" % ((levene[1] >.05), levene))


if levene[1]>.05:
	cor = stats.f_oneway(df['total_score'][(df['conflict'] == "Standoff") & (df['is_pakistan']==True)], 
	             df['total_score'][(df['conflict'] == "Mumbai") & (df['is_pakistan']==True)],
	             df['total_score'][(df['conflict'] == "Burhan") & (df['is_pakistan']==True)],
	             df['total_score'][(df['conflict'] == "Non-conflict") & (df['is_pakistan']==True)])
	tracker.write("\nANOVA results: cor %f| p %f\n" % cor)
else:
	cor = stats.kruskal(df['total_score'][(df['conflict'] == "Standoff") & (df['is_pakistan']==True)], 
	             df['total_score'][(df['conflict'] == "Mumbai") & (df['is_pakistan']==True)],
	             df['total_score'][(df['conflict'] == "Burhan") & (df['is_pakistan']==True)],
	             df['total_score'][(df['conflict'] == "Non-conflict") & (df['is_pakistan']==True)])
	tracker.write("\nKruskal Wallis results: cor %f| p %f\n" % cor)
	
tracker.write("\nConover Post Hoc Test\n")
sp.posthoc_conover(df[df["is_pakistan"]==True], val_col='total_score', group_col='conflict').to_csv(tracker, mode="a")


#HYPOTHESIS 4 Kashmir election to Kash non election
tracker.write("\n\n\n\nHYPOTHESIS 4: Kashmir-related headlines in election periods will have more negative sentiment scores on average than Kashmir-related headlines during non-election periods\r\n")

rp.summary_cont(df[df['is_kashmir'] == True].groupby(['election'])['total_score']).to_csv(tracker, mode="a")
tracker.write("Kashmir - Election Summary\n")
kedf = df[(df['election']==True)& (df['is_kashmir']==True)]
rp.summary_cont(kedf.groupby(['year'])['total_score']).to_csv(tracker, mode="a")

tracker.write("Kashmir - Non Election Summary\n")
knedf = df[(df['election']==False) & (df['is_kashmir']==True)]
rp.summary_cont(knedf.groupby(['year'])['total_score']).to_csv(tracker, mode="a")

levene = stats.levene(kedf['total_score'], knedf["total_score"])
tracker.write("\nVariance is equal: %r, %s\n" % ((levene[1] >.05), levene))
cor = stats.ttest_ind(kedf['total_score'], knedf['total_score'], equal_var=(levene[1] >.05))
tracker.write("\nANOVA results overal Kashmir election: cor %f| p %f\n" % cor)

data = [df['total_score'][(df['year'] == y) & (df['is_kashmir']==tv)] for tv in [True, False] for y in range(2001, 2019)]
cor = stats.kruskal(*data)
tracker.write("\nKruskal Wallis results kashmir y-to-y: cor %f| p %f\n" % cor)
sp.posthoc_conover(df[df["is_kashmir"]==True], val_col='total_score', group_col='year').to_csv(tracker, mode='a')

#HYPOTHESIS 5 Pak election to Pak non election
tracker.write("\n\n\n\nHYPOTHESIS 5: Pakistan-related headlines in election periods will have more negative sentiment scores on average than Pakistan-related headlines during non-election periods\r\n")
rp.summary_cont(df[df['is_pakistan'] == True].groupby(['election'])['total_score']).to_csv(tracker, mode="a")
tracker.write("\nPakistan - Election Summary\n")
pedf = df[(df['election']==True)& (df['is_pakistan']==True)]
rp.summary_cont(pedf.groupby(['year'])['total_score']).to_csv(tracker, mode="a")

tracker.write("\nPakistan - Non Election Summary\n")
pnedf = df[(df['election']==False) & (df['is_pakistan']==True)]
rp.summary_cont(pnedf.groupby(['year'])['total_score']).to_csv(tracker, mode="a")

levene = stats.levene(pedf['total_score'], pnedf["total_score"])
tracker.write("\nVariance is equal: %r, %s\n" % ((levene[1] >.05), levene))
cor = stats.ttest_ind(pedf['total_score'], pnedf['total_score'], equal_var=(levene[1] >.05))
tracker.write("\nANOVA results overall Pakistan election: cor %f| p %f\n" % cor)

data = [df['total_score'][(df['year'] == y) & (df['is_pakistan']==tv)] for tv in [True, False] for y in range(2001, 2019)]
cor = stats.kruskal(*data)
tracker.write("\nKruskal Wallis results Pakistan y-to-y: cor %f| p %f\n" % cor)
sp.posthoc_conover(df[df["is_pakistan"]==True], val_col='total_score', group_col='year').to_csv(tracker, mode='a')

#HYPOTHESIS 6 Kash affect to Kash affect in conflict
tracker.write("\n\n\n\nHYPOTHESIS 6\n")
kec = df[df["is_kashmir"]==True]
#kcs = kec.groupby(['emotion', 'conflict'])
#print(rp.summary_cont(kcs))
crosstab = pd.crosstab(kec['conflict'], kec['emotion'])
crosstab.to_csv(tracker, mode="a")

chi2, p, dof, expected = stats.chi2_contingency(crosstab)
tracker.write(f" Compute chi square\nChi2 value= {chi2}\np-value= {p}\nDegrees of freedom= {dof}\n")
tracker.write("\nBonferroni-adjusted method\n")
dummies = pd.get_dummies(kec['conflict'])
for series in dummies:
    crosstab = pd.crosstab(dummies[f"{series}"], kec['emotion'])
    crosstab.to_csv(tracker, mode="a")
    chi2, p, dof, expected = stats.chi2_contingency(crosstab)
    tracker.write(f"Chi2 value= {chi2}\np-value= {p}\nDegrees of freedom= {dof}\n")
#print(stats.chisquare(kcs.size().divide(len(df)).multiply(100)))

#HYPOTHESIS 7 Pak affect to Pak affect in conflict
tracker.write("\n\n\n\nHYPOTHESIS 7\n")
pec = df[df["is_pakistan"]==True]

crosstab = pd.crosstab(pec['conflict'], pec['emotion'])
crosstab.to_csv(tracker, mode="a")

chi2, p, dof, expected = stats.chi2_contingency(crosstab)
tracker.write(f" Compute chi square\nChi2 value= {chi2}\n-value= {p}\nDegrees of freedom= {dof}\n")
tracker.write("\nBonferroni-adjusted method\n")
dummies = pd.get_dummies(pec['conflict'])
for series in dummies:
    crosstab = pd.crosstab(dummies[f"{series}"], pec['emotion'])
    crosstab.to_csv(tracker, mode="a")
    chi2, p, dof, expected = stats.chi2_contingency(crosstab)
    tracker.write(f"Chi2 value= {chi2}\np-value= {p}\nDegrees of freedom= {dof}\n\n")
#print(stats.chisquare(pcs.size().divide(len(df)).multiply(100)))


