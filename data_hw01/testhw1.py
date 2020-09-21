#-- GEO1001.2020--hw01
#-- [GIORGOS TRIANTAFYLLOU] 
#-- [5381738]

import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.stats
import xlrd
import xlsxwriter

#lesson A1
# Compute mean statistics 
# (mean, variance and standard deviation for each
#  of the sensors variables), what do you observe from 
# the results?
dfA = pd.read_csv("HEAT-A_final.csv" , skiprows = 5, sep=";",encoding="UTF-8",names=[x for x in range(19)])
dfB = pd.read_csv("HEAT-B_final.csv" , skiprows = 5, sep=";",encoding="UTF-8",names=[x for x in range(19)])
dfC = pd.read_csv("HEAT-C_final.csv" , skiprows = 5, sep=";",encoding="UTF-8",names=[x for x in range(19)])
dfD = pd.read_csv("HEAT-D_final.csv" , skiprows = 5, sep=";",encoding="UTF-8",names=[x for x in range(19)])
dfE = pd.read_csv("HEAT-E_final.csv" , skiprows = 5, sep=";",encoding="UTF-8",names=[x for x in range(19)])

MEAN_A = dfA.mean()
VAR_A = dfA.var()
STD_A = dfA.std()
MEAN_A = MEAN_A.to_numpy()
VAR_A = VAR_A.to_numpy()
STD_A = STD_A.to_numpy()

MEAN_B = dfB.mean()
VAR_B = dfB.var()
STD_B = dfB.std()
MEAN_B = MEAN_B.to_numpy()
VAR_B = VAR_B.to_numpy()
STD_B = STD_B.to_numpy()

MEAN_C = dfC.mean()
VAR_C = dfC.var()
STD_C = dfC.std()
MEAN_C = MEAN_C.to_numpy()
VAR_C = VAR_C.to_numpy()
STD_C = STD_C.to_numpy()

MEAN_D = dfD.mean()
VAR_D = dfD.var()
STD_D = dfD.std()
MEAN_D = MEAN_D.to_numpy()
VAR_D = VAR_D.to_numpy()
STD_D = STD_D.to_numpy()

MEAN_E = dfE.mean()
VAR_E = dfE.var()
STD_E = dfE.std()
MEAN_E = MEAN_E.to_numpy()
VAR_E = VAR_E.to_numpy()
STD_E = STD_E.to_numpy()

# Create 1 plot that contains histograms
# for the 5 sensors Temperature values. 
# Compare histograms with 5 and 50 bins, 
# why is the number of bins important?
temp_A=dfA[4]
temp_B=dfB[4]
temp_C=dfC[4]
temp_D=dfD[4]
temp_E=dfE[4]
fig1 = plt.figure(1)
plt.hist([temp_A,temp_B,temp_C,temp_D,temp_E], bins=5, label=['HEAT A','HEAT B','HEAT C','HEAT D','HEAT E'], density=True)
plt.xlabel('Temperature')
plt.ylabel('Number of timeperiods')
plt.title('Temperature Histogram from 5 sensors with 5 bins')
plt.legend(loc='upper right')
fig2 = plt.figure(2)
plt.hist([temp_A,temp_B,temp_C,temp_D,temp_E], bins=50, label=['HEAT A','HEAT B','HEAT C','HEAT D','HEAT E'], density=True)
plt.xlabel('Temperature')
plt.ylabel('Number of timeperiods')
plt.title('Temperature Histogram from 5 sensors with 50 bins')
plt.legend(loc='upper right')

# Create 1 plot where frequency poligons for the 5 sensors 
# Temperature values overlap in different colors with a legend.
fig3 = plt.figure(3)
y_A,edges = np.histogram(temp_A,bins=27)
centers = 0.5*(edges[1:]+ edges[:-1])
plt.plot(centers,y_A,'-*',label=('Sensor A'))
y_B,edges = np.histogram(temp_B,bins=27)
plt.plot(centers,y_B,'-*',label=('Sensor B'))
y_C,edges = np.histogram(temp_C,bins=27)
plt.plot(centers,y_C,'-*',label=('Sensor C'))
y_D,edges = np.histogram(temp_D,bins=27)
plt.plot(centers,y_D,'-*',label=('Sensor D'))
y_E,edges = np.histogram(temp_E,bins=27)
plt.plot(centers,y_E,'-*',label=('Sensor E'))
plt.xlabel('Temperature')
plt.ylabel('Number of timeperiods')
plt.title('Frequency Poligons of 5 sensors')
plt.legend(loc='upper right')

# Generate 3 plots that include the 5 sensors boxplot for:
# Wind Speed, Wind Direction and Temperature.

#boxplots for Wind Direction
WDA=dfA[0]
WDB=dfB[0]
WDC=dfC[0]
WDD=dfD[0]
WDE=dfE[0]
#boxplots for Wind Speed in m/s
WSA=dfA[1]
WSB=dfB[1]
WSC=dfC[1]
WSD=dfD[1]
WSE=dfE[1]
fig4=plt.figure(4)
plt.boxplot([temp_A,temp_B,temp_C,temp_D,temp_E], showmeans=True, labels=['Sensor A','Sensor B','Sensor C','Sensor D','Sensor e'])
plt.ylabel('Temperature Values')
plt.title('Temperature Boxplot')
fig5=plt.figure(5)
plt.boxplot ([WDA,WDB,WDC,WDD,WDE],showmeans=True, labels=['Sensor A','Sensor B','Sensor C','Sensor D','Sensor e'])
plt.ylabel('Wind Direction Values')
plt.title('Wind Direction Boxplot')
fig6=plt.figure(6)
plt.boxplot ([WSA,WSB,WSC,WSD,WSE],showmeans=True, labels=['Sensor A','Sensor B','Sensor C','Sensor D','Sensor e'])
plt.ylabel('Wind Speed Values')
plt.title('Wind Speed Boxplot')

#lesson A2
# Plot PMF, PDF and CDF for the 5 sensors 
# Temperature values. 
# Describe the behaviour of the distributions, 
# are they all similar? what about their tails?

#compute and plot pmf
def pmf(sample):
 	c = sample.value_counts()
 	p = c/len(sample)
 	return p 

fig7, axs = plt.subplots(5)
fig7.suptitle("PMF for temperature values")
df_a = pmf(temp_A)
c_a = df_a.sort_index()

df_b = pmf(temp_B)
c_b = df_b.sort_index()

df_c = pmf(temp_C)
c_c = df_c.sort_index()

df_d = pmf(temp_D)
c_d = df_d.sort_index()

df_e = pmf(temp_E)
c_e = df_e.sort_index()

axs[0].bar(c_a.index, c_a)
axs[1].bar(c_b.index, c_b)
axs[2].bar(c_c.index, c_c)
axs[3].bar(c_d.index, c_d)
axs[4].bar(c_e.index, c_e)

# plot pdf
fig8, axs = plt.subplots(5)
fig8.suptitle("PDF for temperature values")
sns.distplot([temp_A.astype(float)], ax=axs[0])
sns.distplot([temp_B.astype(float)], ax=axs[1])
sns.distplot([temp_C.astype(float)], ax=axs[2])
sns.distplot([temp_D.astype(float)], ax=axs[3])
sns.distplot([temp_E.astype(float)], ax=axs[4])
axs[0].hist([temp_A.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)
axs[1].hist([temp_B.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)
axs[2].hist([temp_C.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)
axs[3].hist([temp_D.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)
axs[4].hist([temp_E.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)


# plot cdf
fig9, axs = plt.subplots(5)
fig9.suptitle("CDF for temperature")
a1=axs[0].hist([temp_A.astype(float)],bins=27, cumulative=True, alpha=0.7, rwidth=0.85)
a2=axs[1].hist([temp_B.astype(float)],bins=27, cumulative=True, alpha=0.7, rwidth=0.85)
a3=axs[2].hist([temp_C.astype(float)],bins=27, cumulative=True, alpha=0.7, rwidth=0.85)
a4=axs[3].hist([temp_D.astype(float)],bins=27, cumulative=True, alpha=0.7, rwidth=0.85)
a5=axs[4].hist([temp_E.astype(float)],bins=27, cumulative=True, alpha=0.7, rwidth=0.85)
axs[0].plot(a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0], color='k')
axs[1].plot(a2[1][1:]-(a2[1][1:]-a2[1][:-1])/2,a2[0], color='k')
axs[2].plot(a3[1][1:]-(a3[1][1:]-a3[1][:-1])/2,a3[0], color='k')
axs[3].plot(a4[1][1:]-(a4[1][1:]-a4[1][:-1])/2,a4[0], color='k')
axs[4].plot(a5[1][1:]-(a5[1][1:]-a5[1][:-1])/2,a5[0], color='k')

# For the Wind Speed values, 
# plot the pdf and the kernel density estimation. 
# Comment the differences.

# Plot Kernel for Wind Speed
fig14=plt.figure(14)
kernel = stats.gaussian_kde(WSA)
ax = fig14.add_subplot(111)
x_eval = np.linspace(-10, 10, num=200)
ax.plot(x_eval,kernel(x_eval))
sns.distplot([WSA.astype(float)])

fig15=plt.figure(15)
kernel = stats.gaussian_kde(WSB)
ax = fig15.add_subplot(111)
x_eval = np.linspace(-10, 10, num=200)
ax.plot(x_eval,kernel(x_eval))
sns.distplot([WSB.astype(float)])

fig16=plt.figure(16)
kernel = stats.gaussian_kde(WSC)
ax = fig16.add_subplot(111)
x_eval = np.linspace(-10, 10, num=200)
ax.plot(x_eval,kernel(x_eval))
sns.distplot([WSC.astype(float)])

fig17=plt.figure(17)
kernel = stats.gaussian_kde(WSD)
ax = fig17.add_subplot(111)
x_eval = np.linspace(-10, 10, num=200)
ax.plot(x_eval,kernel(x_eval))
sns.distplot([WSD.astype(float)])

fig18=plt.figure(18)
kernel = stats.gaussian_kde(WSE)
ax = fig18.add_subplot(111)
x_eval = np.linspace(-10, 10, num=200)
ax.plot(x_eval,kernel(x_eval))
sns.distplot([WSE.astype(float)])

# plot pdf for Wind Speed
fig19 = plt.figure(19)
# ax1 = fig.add_subplot(111)
a1=plt.hist([WSA.astype(float),WSB.astype(float),WSC.astype(float),WSD.astype(float),WSE.astype(float)], density=True, alpha=0.7, rwidth=0.85)
sns.distplot(WSA.astype(float))
sns.distplot(WSB.astype(float))
sns.distplot(WSC.astype(float))
sns.distplot(WSD.astype(float))
sns.distplot(WSE.astype(float))

#lesson A3
#Compute the correlations between all the sensors for the 
# variables: Temperature, Wet Bulb Globe, Crosswind Speed. 
# Perform correlation between sensors with the same variable, not between two different variables; 
# for example, correlate Temperature time series between sensor A and B.
# Use Pearson’s and Spearmann’s rank coefficients. 
# Make a scatter plot with both coefficients with the 3 variables.
#Crosswind Speed
CRWA=dfA[2]
CRWB=dfB[2]
CRWC=dfC[2]
CRWD=dfD[2]
CRWE=dfE[2]

#Wet Bulb Globe
WBGTA=dfA[16]
WBGTB=dfB[16]
WBGTC=dfC[16]
WBGTD=dfD[16]
WBGTE=dfE[16]

# Heat Map A - Heat Map B

temp_A_interp = np.interp(np.linspace(0,len(temp_B),len(temp_B)), np.linspace(0,len(temp_A),len(temp_A)),temp_A)
crw_A_interp = np.interp(np.linspace(0,len(CRWB),len(CRWB)), np.linspace(0,len(CRWA),len(CRWA)),CRWA)
wbgt_A_interp = np.interp(np.linspace(0,len(WBGTB),len(WBGTB)), np.linspace(0,len(WBGTA),len(WBGTA)),WBGTA)

pcoef_temp_AB = stats.pearsonr(temp_A_interp,temp_B)
prcoef_temp_AB = stats.spearmanr(temp_A_interp,temp_B)

pcoef_CRW_AB = stats.pearsonr(crw_A_interp,CRWB)
prcoef_CRW_AB = stats.spearmanr(crw_A_interp,CRWB)

pcoef_WBGT_AB = stats.pearsonr(wbgt_A_interp,WBGTB)
prcoef_WBGT_AB = stats.spearmanr(wbgt_A_interp,WBGTB)

# Heat Map A - Heat Map C
temp_A_interp = np.interp(np.linspace(0,len(temp_C),len(temp_C)), np.linspace(0,len(temp_A),len(temp_A)),temp_A)
crw_A_interp = np.interp(np.linspace(0,len(CRWC),len(CRWC)), np.linspace(0,len(CRWA),len(CRWA)),CRWA)
wbgt_A_interp = np.interp(np.linspace(0,len(WBGTC),len(WBGTC)), np.linspace(0,len(WBGTA),len(WBGTA)),WBGTA)

pcoef_temp_AC = stats.pearsonr(temp_A_interp,temp_C)
prcoef_temp_AC = stats.spearmanr(temp_A_interp,temp_C)

pcoef_CRW_AC = stats.pearsonr(crw_A_interp,CRWC)
prcoef_CRW_AC = stats.spearmanr(crw_A_interp,CRWC)

pcoef_WBGT_AC = stats.pearsonr(wbgt_A_interp,WBGTC)
prcoef_WBGT_AC = stats.spearmanr(wbgt_A_interp,WBGTC)

# Heat Map A - Heat Map D
temp_A_interp = np.interp(np.linspace(0,len(temp_D),len(temp_D)), np.linspace(0,len(temp_A),len(temp_A)),temp_A)
crw_A_interp = np.interp(np.linspace(0,len(CRWD),len(CRWD)), np.linspace(0,len(CRWA),len(CRWA)),CRWA)
wbgt_A_interp = np.interp(np.linspace(0,len(WBGTD),len(WBGTD)), np.linspace(0,len(WBGTA),len(WBGTA)),WBGTA)

pcoef_temp_AD = stats.pearsonr(temp_A_interp,temp_D)
prcoef_temp_AD = stats.spearmanr(temp_A_interp,temp_D)

pcoef_CRW_AD = stats.pearsonr(crw_A_interp,CRWD)
prcoef_CRW_AD = stats.spearmanr(crw_A_interp,CRWD)

pcoef_WBGT_AD = stats.pearsonr(wbgt_A_interp,WBGTD)
prcoef_WBGT_AD = stats.spearmanr(wbgt_A_interp,WBGTD)
# Heat Map A - Heat Map E
temp_A_interp = np.interp(np.linspace(0,len(temp_E),len(temp_E)), np.linspace(0,len(temp_A),len(temp_A)),temp_A)
crw_A_interp = np.interp(np.linspace(0,len(CRWE),len(CRWE)), np.linspace(0,len(CRWA),len(CRWA)),CRWA)
wbgt_A_interp = np.interp(np.linspace(0,len(WBGTE),len(WBGTE)), np.linspace(0,len(WBGTA),len(WBGTA)),WBGTA)

pcoef_temp_AE = stats.pearsonr(temp_A_interp,temp_E)
prcoef_temp_AE = stats.spearmanr(temp_A_interp,temp_E)

pcoef_CRW_AE = stats.pearsonr(crw_A_interp,CRWE)
prcoef_CRW_AE = stats.spearmanr(crw_A_interp,CRWE)

pcoef_WBGT_AE = stats.pearsonr(wbgt_A_interp,WBGTE)
prcoef_WBGT_AE = stats.spearmanr(wbgt_A_interp,WBGTE)
# Heat Map B - Heat Map C
temp_B_interp = np.interp(np.linspace(0,len(temp_C),len(temp_C)), np.linspace(0,len(temp_B),len(temp_B)),temp_B)
crw_B_interp = np.interp(np.linspace(0,len(CRWC),len(CRWC)), np.linspace(0,len(CRWB),len(CRWB)),CRWB)
wbgt_B_interp = np.interp(np.linspace(0,len(WBGTC),len(WBGTC)), np.linspace(0,len(WBGTB),len(WBGTB)),WBGTB)

pcoef_temp_BC = stats.pearsonr(temp_B_interp,temp_C)
prcoef_temp_BC = stats.spearmanr(temp_B_interp,temp_C)

pcoef_CRW_BC = stats.pearsonr(crw_B_interp,CRWC)
prcoef_CRW_BC = stats.spearmanr(crw_B_interp,CRWC)

pcoef_WBGT_BC = stats.pearsonr(wbgt_B_interp,WBGTC)
prcoef_WBGT_BC = stats.spearmanr(wbgt_B_interp,WBGTC)
# Heat Map B - Heat Map D
temp_B_interp = np.interp(np.linspace(0,len(temp_D),len(temp_D)), np.linspace(0,len(temp_B),len(temp_B)),temp_B)
crw_B_interp = np.interp(np.linspace(0,len(CRWD),len(CRWD)), np.linspace(0,len(CRWB),len(CRWB)),CRWB)
wbgt_B_interp = np.interp(np.linspace(0,len(WBGTD),len(WBGTD)), np.linspace(0,len(WBGTB),len(WBGTB)),WBGTB)

pcoef_temp_BD = stats.pearsonr(temp_B_interp,temp_D)
prcoef_temp_BD = stats.spearmanr(temp_B_interp,temp_D)

pcoef_CRW_BD = stats.pearsonr(crw_B_interp,CRWD)
prcoef_CRW_BD = stats.spearmanr(crw_B_interp,CRWD)

pcoef_WBGT_BD = stats.pearsonr(wbgt_B_interp,WBGTD)
prcoef_WBGT_BD = stats.spearmanr(wbgt_B_interp,WBGTD)
# Heat Map B - Heat Map E
temp_B_interp = np.interp(np.linspace(0,len(temp_E),len(temp_E)), np.linspace(0,len(temp_B),len(temp_B)),temp_B)
crw_B_interp = np.interp(np.linspace(0,len(CRWE),len(CRWE)), np.linspace(0,len(CRWB),len(CRWB)),CRWB)
wbgt_B_interp = np.interp(np.linspace(0,len(WBGTE),len(WBGTE)), np.linspace(0,len(WBGTB),len(WBGTB)),WBGTB)

pcoef_temp_BE = stats.pearsonr(temp_B_interp,temp_E)
prcoef_temp_BE = stats.spearmanr(temp_B_interp,temp_E)

pcoef_CRW_BE = stats.pearsonr(crw_B_interp,CRWE)
prcoef_CRW_BE = stats.spearmanr(crw_B_interp,CRWE)

pcoef_WBGT_BE = stats.pearsonr(wbgt_B_interp,WBGTE)
prcoef_WBGT_BE = stats.spearmanr(wbgt_B_interp,WBGTE)
# Heat Map C - Heat Map D
temp_C_interp = np.interp(np.linspace(0,len(temp_D),len(temp_D)), np.linspace(0,len(temp_C),len(temp_C)),temp_C)
crw_C_interp = np.interp(np.linspace(0,len(CRWD),len(CRWD)), np.linspace(0,len(CRWC),len(CRWC)),CRWC)
wbgt_C_interp = np.interp(np.linspace(0,len(WBGTD),len(WBGTD)), np.linspace(0,len(WBGTC),len(WBGTC)),WBGTC)

pcoef_temp_CD = stats.pearsonr(temp_C_interp,temp_D)
prcoef_temp_CD = stats.spearmanr(temp_C_interp,temp_D)

pcoef_CRW_CD = stats.pearsonr(crw_C_interp,CRWD)
prcoef_CRW_CD = stats.spearmanr(crw_C_interp,CRWD)

pcoef_WBGT_CD = stats.pearsonr(wbgt_C_interp,WBGTD)
prcoef_WBGT_CD = stats.spearmanr(wbgt_C_interp,WBGTD)
# Heat Map C - Heat Map E
temp_C_interp = np.interp(np.linspace(0,len(temp_E),len(temp_E)), np.linspace(0,len(temp_C),len(temp_C)),temp_C)
crw_C_interp = np.interp(np.linspace(0,len(CRWE),len(CRWE)), np.linspace(0,len(CRWC),len(CRWC)),CRWC)
wbgt_C_interp = np.interp(np.linspace(0,len(WBGTE),len(WBGTE)), np.linspace(0,len(WBGTC),len(WBGTC)),WBGTC)

pcoef_temp_CE = stats.pearsonr(temp_C_interp,temp_E)
prcoef_temp_CE = stats.spearmanr(temp_C_interp,temp_E)

pcoef_CRW_CE = stats.pearsonr(crw_C_interp,CRWE)
prcoef_CRW_CE = stats.spearmanr(crw_C_interp,CRWE)

pcoef_WBGT_CE = stats.pearsonr(wbgt_C_interp,WBGTE)
prcoef_WBGT_CE = stats.spearmanr(wbgt_C_interp,WBGTE)
# Heat Map D - Heat Map E
temp_D_interp = np.interp(np.linspace(0,len(temp_E),len(temp_E)), np.linspace(0,len(temp_D),len(temp_D)),temp_D)
crw_D_interp = np.interp(np.linspace(0,len(CRWE),len(CRWE)), np.linspace(0,len(CRWD),len(CRWD)),CRWD)
wbgt_D_interp = np.interp(np.linspace(0,len(WBGTE),len(WBGTE)), np.linspace(0,len(WBGTD),len(WBGTD)),WBGTD)

pcoef_temp_DE = stats.pearsonr(temp_D_interp,temp_E)
prcoef_temp_DE = stats.spearmanr(temp_D_interp,temp_E)

pcoef_CRW_DE = stats.pearsonr(crw_D_interp,CRWE)
prcoef_CRW_DE = stats.spearmanr(crw_D_interp,CRWE)

pcoef_WBGT_DE = stats.pearsonr(wbgt_D_interp,WBGTE)
prcoef_WBGT_DE = stats.spearmanr(wbgt_D_interp,WBGTE)

Workbook=xlsxwriter.Workbook("Correlation.xlsx")
Worksheet=Workbook.add_worksheet()
Worksheet.write("B1","Pearson Coefficient")
Worksheet.write("C1","Spearman Coefficient")
# Worksheet.write("B2",pcoef_temp_AB[0])
Worksheet.write("B3",pcoef_CRW_AB[0])
Worksheet.write("B4",pcoef_WBGT_AB[0])

Worksheet.write("C2",prcoef_temp_AB[0])
Worksheet.write("C3",prcoef_CRW_AB[0])
Worksheet.write("C4",prcoef_WBGT_AB[0])

Worksheet.write("B5",pcoef_temp_AC[0])
Worksheet.write("B6",pcoef_CRW_AC[0])
Worksheet.write("B7",pcoef_WBGT_AC[0])

Worksheet.write("C5",prcoef_temp_AC[0])
Worksheet.write("C6",prcoef_CRW_AC[0])
Worksheet.write("C7",prcoef_WBGT_AC[0])

Worksheet.write("B8",pcoef_temp_AD[0])
Worksheet.write("B9",pcoef_CRW_AD[0])
Worksheet.write("B10",pcoef_WBGT_AD[0])

Worksheet.write("C8",prcoef_temp_AD[0])
Worksheet.write("C9",prcoef_CRW_AD[0])
Worksheet.write("C10",prcoef_WBGT_AD[0])

Worksheet.write("B11",pcoef_temp_AE[0])
Worksheet.write("B12",pcoef_CRW_AE[0])
Worksheet.write("B13",pcoef_WBGT_AE[0])

Worksheet.write("C11",prcoef_temp_AE[0])
Worksheet.write("C12",prcoef_CRW_AE[0])
Worksheet.write("C13",prcoef_WBGT_AE[0])

Worksheet.write("B14",pcoef_temp_BC[0])
Worksheet.write("B15",pcoef_CRW_BC[0])
Worksheet.write("B16",pcoef_WBGT_BC[0])

Worksheet.write("C14",prcoef_temp_BC[0])
Worksheet.write("C15",prcoef_CRW_BC[0])
Worksheet.write("C16",prcoef_WBGT_BC[0])

Worksheet.write("B17",pcoef_temp_BD[0])
Worksheet.write("B18",pcoef_CRW_BD[0])
Worksheet.write("B19",pcoef_WBGT_BD[0])

Worksheet.write("C17",prcoef_temp_BD[0])
Worksheet.write("C18",prcoef_CRW_BD[0])
Worksheet.write("C19",prcoef_WBGT_BD[0])

Worksheet.write("B20",pcoef_temp_BE[0])
Worksheet.write("B21",pcoef_CRW_BE[0])
Worksheet.write("B22",pcoef_WBGT_BE[0])

Worksheet.write("C20",prcoef_temp_BE[0])
Worksheet.write("C21",prcoef_CRW_BE[0])
Worksheet.write("C22",prcoef_WBGT_BE[0])

Worksheet.write("B23",pcoef_temp_CD[0])
Worksheet.write("B24",pcoef_CRW_CD[0])
Worksheet.write("B25",pcoef_WBGT_CD[0])

Worksheet.write("C23",prcoef_temp_CD[0])
Worksheet.write("C24",prcoef_CRW_CD[0])
Worksheet.write("C25",prcoef_WBGT_CD[0])

Worksheet.write("B26",pcoef_temp_CE[0])
Worksheet.write("B27",pcoef_CRW_CE[0])
Worksheet.write("B28",pcoef_WBGT_CE[0])

Worksheet.write("C26",prcoef_temp_CE[0])
Worksheet.write("C27",prcoef_CRW_CE[0])
Worksheet.write("C28",prcoef_WBGT_CE[0])

Worksheet.write("B29",pcoef_temp_DE[0])
Worksheet.write("B30",pcoef_CRW_DE[0])
Worksheet.write("B31",pcoef_WBGT_DE[0])

Worksheet.write("C29",prcoef_temp_DE[0])
Worksheet.write("C30",prcoef_CRW_DE[0])
Worksheet.write("C31",prcoef_WBGT_DE[0])
Workbook.close()

#Pearson and Spearman Coefficient for Temperature

Pearson = [pcoef_temp_AB[0], pcoef_temp_AC[0], pcoef_temp_AD[0], pcoef_temp_AE[0], pcoef_temp_BC[0], pcoef_temp_BD[0], pcoef_temp_BE[0], pcoef_temp_CD[0], pcoef_temp_CE[0], pcoef_temp_DE[0]]
Spearman = [prcoef_temp_AB[0], prcoef_temp_AC[0], prcoef_temp_AD[0], prcoef_temp_AE[0], prcoef_temp_BC[0], prcoef_temp_BD[0], prcoef_temp_BE[0], prcoef_temp_CD[0], prcoef_temp_CE[0], prcoef_temp_DE[0]]

A1=[1,2,3,4,5,6,7,8,9,10]
Labels=["AB","AC","AD","AE","BC","BD","BE","CD","CE","DE"]

fig16= plt.figure(16)
plt.title("Temperature Correlations with Pearson and Spearman coefficient")
plt.xticks(A1,Labels)
Pearson_scatter=plt.scatter(A1,Pearson)
Spearman_scatter=plt.scatter(A1,Spearman)
plt.legend((Pearson_scatter,Spearman_scatter),('Pearson Coefficient','Spearman Coefficient'), loc='upper right')


# Step 4 --- scatter plot dimensional variables
# fig = plt.figure(figsize=(17,5))
# ax1 = fig.add_subplot(111)
# ax1.scatter(heights1,weights,c='b')
# ax1.set_xlabel('Heights [cm]')
# ax1.set_ylabel('Weights [kg]')
# plt.show()

#lesson A4
fig19, axs = plt.subplots(5)
fig9.suptitle("CDF for Wind Speed")
a1=axs[0].hist([WSA.astype(float)],bins=27, cumulative=True, alpha=0.7, rwidth=0.85)
a2=axs[1].hist([WSB.astype(float)],bins=27, cumulative=True, alpha=0.7, rwidth=0.85)
a3=axs[2].hist([WSC.astype(float)],bins=27, cumulative=True, alpha=0.7, rwidth=0.85)
a4=axs[3].hist([WSD.astype(float)],bins=27, cumulative=True, alpha=0.7, rwidth=0.85)
a5=axs[4].hist([WSE.astype(float)],bins=27, cumulative=True, alpha=0.7, rwidth=0.85)
axs[0].plot(a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0], color='k')
axs[1].plot(a2[1][1:]-(a2[1][1:]-a2[1][:-1])/2,a2[0], color='k')
axs[2].plot(a3[1][1:]-(a3[1][1:]-a3[1][:-1])/2,a3[0], color='k')
axs[3].plot(a4[1][1:]-(a4[1][1:]-a4[1][:-1])/2,a4[0], color='k')
axs[4].plot(a5[1][1:]-(a5[1][1:]-a5[1][:-1])/2,a5[0], color='k')

# def mean_confidence_interval(data, confidence=0.95):
# 	a = 1.0 * np.array(data)
# 	n = len(a)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


plt.legend()
plt.show()