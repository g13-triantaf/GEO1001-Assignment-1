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
statistics_HEAT_A=([[MEAN_A],[VAR_A],[STD_A]])
print(statistics_HEAT_A)
MEAN_B = dfB.mean()
VAR_B = dfB.var()
STD_B = dfB.std()
statistics_HEAT_B=([[MEAN_B],[VAR_B],[STD_B]])
print(statistics_HEAT_B)
MEAN_C = dfC.mean()
VAR_C = dfC.var()
STD_C = dfC.std()
statistics_HEAT_C=([[MEAN_C],[VAR_C],[STD_C]])
print(statistics_HEAT_C)
MEAN_D = dfD.mean()
VAR_D = dfD.var()
STD_D = dfD.std()
statistics_HEAT_D=([[MEAN_D],[VAR_D],[STD_D]])
print(statistics_HEAT_D)
MEAN_E = dfE.mean()
VAR_E = dfE.var()
STD_E = dfE.std()
statistics_HEAT_E=([[MEAN_E],[VAR_E],[STD_E]])
print(statistics_HEAT_E)
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
plt.xlabel('Temperature in °C')
plt.ylabel('Frequency')
plt.title('Temperature Histogram from 5 sensors with 5 bins')
plt.legend(loc='upper right')
fig2 = plt.figure(2)
plt.hist([temp_A,temp_B,temp_C,temp_D,temp_E], bins=50, label=['HEAT A','HEAT B','HEAT C','HEAT D','HEAT E'], density=True)
plt.xlabel('Temperature in °C')
plt.ylabel('Frequency')
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
plt.xlabel('Temperature in °C')
plt.ylabel('Frequency')
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
plt.ylabel('Temperature in °C')
plt.title('Temperature Boxplot')
fig5=plt.figure(5)
plt.boxplot ([WDA,WDB,WDC,WDD,WDE],showmeans=True, labels=['Sensor A','Sensor B','Sensor C','Sensor D','Sensor e'])
plt.ylabel('Wind Direction')
plt.title('Wind Direction Boxplot')
fig6=plt.figure(6)
plt.boxplot ([WSA,WSB,WSC,WSD,WSE],showmeans=True, labels=['Sensor A','Sensor B','Sensor C','Sensor D','Sensor e'])
plt.ylabel('Wind Speed [m/s]')
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

fig7, axs = plt.subplots(5, sharex=True)
fig7.suptitle("PMF for Temperature")
plt.xlabel('Temperature in °C')
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
fig8, axs = plt.subplots(5, sharex=True)
fig8.suptitle("PDF for Temperature")
plt.xlabel('Temperature in °C')
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
fig9, axs = plt.subplots(5, sharex=True)
fig9.suptitle("CDF for Temperature")
plt.xlabel('Temperature in °C')
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

#Plot Kernel for Wind Speed

#Sensor A
fig10, axs = plt.subplots(2,sharey=True, sharex=True)
axs[0].title.set_text("PDF for Wind Speed (Sensor A)")
axs[0].hist([WSA.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)
sns.distplot([WSA.astype(float)],ax=axs[0])
kernel = stats.gaussian_kde(WSA)
x_eval = np.linspace(-10, 10, num=200)
axs[1].plot(x_eval,kernel(x_eval))
axs[1].title.set_text('Kernel Density Estimation of Wind Speed (Sensor A)')

#Sensor B
fig11, axs = plt.subplots(2,sharey=True, sharex=True)
axs[0].title.set_text("PDF for Wind Speed (Sensor B)")
axs[0].hist([WSB.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)
sns.distplot([WSB.astype(float)],ax=axs[0])
kernel = stats.gaussian_kde(WSB)
x_eval = np.linspace(-10, 10, num=200)
axs[1].plot(x_eval,kernel(x_eval))
axs[1].title.set_text('Kernel Density Estimation of Wind Speed (Sensor B)')

#Sensor C
fig12, axs = plt.subplots(2,sharey=True, sharex=True)
axs[0].title.set_text("PDF for Wind Speed (Sensor C)")
axs[0].hist([WSC.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)
sns.distplot([WSC.astype(float)],ax=axs[0])
kernel = stats.gaussian_kde(WSC)
x_eval = np.linspace(-10, 10, num=200)
axs[1].plot(x_eval,kernel(x_eval))
axs[1].title.set_text('Kernel Density Estimation of Wind Speed (Sensor C)')

#Sensor D
fig13, axs = plt.subplots(2,sharey=True, sharex=True)
axs[0].title.set_text("PDF for Wind Speed (Sensor D)")
axs[0].hist([WSD.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)
sns.distplot([WSD.astype(float)],ax=axs[0])
kernel = stats.gaussian_kde(WSD)
x_eval = np.linspace(-10, 10, num=200)
axs[1].plot(x_eval,kernel(x_eval))
axs[1].title.set_text('Kernel Density Estimation of Wind Speed (Sensor D)')

#Sensor E
fig14, axs = plt.subplots(2,sharey=True, sharex=True)
axs[0].title.set_text("PDF for Wind Speed (Sensor E)")
axs[0].hist([WSE.astype(float)],bins=27,density=True,alpha=0.7, rwidth=0.85)
sns.distplot([WSE.astype(float)],ax=axs[0])
kernel = stats.gaussian_kde(WSE)
x_eval = np.linspace(-10, 10, num=200)
axs[1].plot(x_eval,kernel(x_eval))
axs[1].title.set_text('Kernel Density Estimation of Wind Speed (Sensor E)')

#lesson A3
# Compute the correlations between all the sensors for the 
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

fig15= plt.figure(15)
plt.title("Temperature Correlations with Pearson and Spearman coefficient")
plt.xticks(A1,Labels)
Pearson_scatter=plt.scatter(A1,Pearson)
Spearman_scatter=plt.scatter(A1,Spearman)
plt.legend((Pearson_scatter,Spearman_scatter),('Pearson Coefficient','Spearman Coefficient'), loc='upper right')

#Pearson and Spearman Coefficient for Crosswind

Pearson = [pcoef_CRW_AB[0], pcoef_CRW_AC[0], pcoef_CRW_AD[0], pcoef_CRW_AE[0], pcoef_CRW_BC[0], pcoef_CRW_BD[0], pcoef_CRW_BE[0], pcoef_CRW_CD[0], pcoef_CRW_CE[0], pcoef_CRW_DE[0]]
Spearman = [prcoef_CRW_AB[0], prcoef_CRW_AC[0], prcoef_CRW_AD[0], prcoef_CRW_AE[0], prcoef_CRW_BC[0], prcoef_CRW_BD[0], prcoef_CRW_BE[0], prcoef_CRW_CD[0], prcoef_CRW_CE[0], prcoef_CRW_DE[0]]

A1=[1,2,3,4,5,6,7,8,9,10]
Labels=["AB","AC","AD","AE","BC","BD","BE","CD","CE","DE"]

fig16= plt.figure(16)
plt.title("Crosswind Correlations with Pearson and Spearman coefficient")
plt.xticks(A1,Labels)
Pearson_scatter=plt.scatter(A1,Pearson)
Spearman_scatter=plt.scatter(A1,Spearman)
plt.legend((Pearson_scatter,Spearman_scatter),('Pearson Coefficient','Spearman Coefficient'), loc='upper right')

#Pearson and Spearman Coefficient for WBGT

Pearson = [pcoef_WBGT_AB[0], pcoef_WBGT_AC[0], pcoef_WBGT_AD[0], pcoef_WBGT_AE[0], pcoef_WBGT_BC[0], pcoef_WBGT_BD[0], pcoef_WBGT_BE[0], pcoef_WBGT_CD[0], pcoef_WBGT_CE[0], pcoef_WBGT_DE[0]]
Spearman = [prcoef_WBGT_AB[0], prcoef_WBGT_AC[0], prcoef_WBGT_AD[0], prcoef_WBGT_AE[0], prcoef_WBGT_BC[0], prcoef_WBGT_BD[0], prcoef_WBGT_BE[0], prcoef_WBGT_CD[0], prcoef_WBGT_CE[0], prcoef_WBGT_DE[0]]

A1=[1,2,3,4,5,6,7,8,9,10]
Labels=["AB","AC","AD","AE","BC","BD","BE","CD","CE","DE"]

fig17= plt.figure(17)
plt.title("WBGT Correlations with Pearson and Spearman coefficient")
plt.xticks(A1,Labels)
Pearson_scatter=plt.scatter(A1,Pearson)
Spearman_scatter=plt.scatter(A1,Spearman)
plt.legend((Pearson_scatter,Spearman_scatter),('Pearson Coefficient','Spearman Coefficient'), loc='upper right')


#lesson A4
fig18, axs = plt.subplots(5, sharex=True, sharey=True)
fig18.suptitle("CDF for Wind Speed")
plt.xlabel("Wind Speed in m/s")
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

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m, m+h

conf_temp_A=(mean_confidence_interval(temp_A,confidence=0.95))
conf_temp_B=(mean_confidence_interval(temp_B,confidence=0.95))
conf_temp_C=(mean_confidence_interval(temp_C,confidence=0.95))
conf_temp_D=(mean_confidence_interval(temp_D,confidence=0.95))
conf_temp_E=(mean_confidence_interval(temp_E,confidence=0.95))

conf_WS_A=(mean_confidence_interval(WSA,confidence=0.95))
conf_WS_B=(mean_confidence_interval(WSB,confidence=0.95))
conf_WS_C=(mean_confidence_interval(WSC,confidence=0.95))
conf_WS_D=(mean_confidence_interval(WSD,confidence=0.95))
conf_WS_E=(mean_confidence_interval(WSE,confidence=0.95))

Workbook=xlsxwriter.Workbook("Confidence_intervals.csv")
Worksheet=Workbook.add_worksheet()
Worksheet.write("A1","m-h")
Worksheet.write("B1","m")
Worksheet.write("C1","m+h")
Worksheet.write("A2",conf_temp_A[0])
Worksheet.write("B2",conf_temp_A[1])
Worksheet.write("C2",conf_temp_A[2])
Worksheet.write("A3",conf_temp_B[0])
Worksheet.write("B3",conf_temp_B[1])
Worksheet.write("C3",conf_temp_B[2])
Worksheet.write("A4",conf_temp_C[0])
Worksheet.write("B4",conf_temp_C[1])
Worksheet.write("C4",conf_temp_C[2])
Worksheet.write("A5",conf_temp_D[0])
Worksheet.write("B5",conf_temp_D[1])
Worksheet.write("C5",conf_temp_D[2])
Worksheet.write("A6",conf_temp_E[0])
Worksheet.write("B6",conf_temp_E[1])
Worksheet.write("C6",conf_temp_E[2])

Worksheet.write("A7",conf_WS_A[0])
Worksheet.write("B7",conf_WS_A[1])
Worksheet.write("C7",conf_WS_A[2])
Worksheet.write("A8",conf_WS_B[0])
Worksheet.write("B8",conf_WS_B[1])
Worksheet.write("C8",conf_WS_B[2])
Worksheet.write("A9",conf_WS_C[0])
Worksheet.write("B9",conf_WS_C[1])
Worksheet.write("C9",conf_WS_C[2])
Worksheet.write("A10",conf_WS_D[0])
Worksheet.write("B10",conf_WS_D[1])
Worksheet.write("C10",conf_WS_D[2])
Worksheet.write("A11",conf_WS_E[0])
Worksheet.write("B11",conf_WS_E[1])
Worksheet.write("C11",conf_WS_E[2])
Workbook.close()

t_ED_temp,p_ED_temp = stats.ttest_ind(temp_E,temp_D)
t_DC_temp,p_DC_temp = stats.ttest_ind(temp_D,temp_C)
t_CB_temp,p_CB_temp = stats.ttest_ind(temp_C,temp_B)
t_BA_temp,p_BA_temp = stats.ttest_ind(temp_B,temp_A)
t_ED_WS,p_ED_WS = stats.ttest_ind(WSE,WSD)
t_DC_WS,p_DC_WS = stats.ttest_ind(WSD,WSC)
t_CB_WS,p_CB_WS = stats.ttest_ind(WSC,WSB)
t_BA_WS,p_BA_WS = stats.ttest_ind(WSB,WSA)

print (t_ED_temp,p_ED_temp)
print (t_DC_temp,p_DC_temp)
print (t_CB_temp,p_CB_temp)
print (t_BA_temp,p_BA_temp)
print (t_ED_WS,p_ED_WS)
print (t_DC_WS,p_DC_WS)
print (t_CB_WS,p_CB_WS)
print (t_BA_WS,p_BA_WS)

#Bonus
data = ([dfA],[dfB],[dfC],[dfD],[dfE])
def average_temperature(data):
    temperature=[data[0:72].mean(),data[72:144].mean(),data[144:216].mean(),data[216:288].mean(),data[288:360].mean(),data[360:432].mean(),data[432:504].mean(),
    data[504:576].mean(),data[576:648].mean(),data[648:720].mean(),data[720:792].mean(),data[792:864].mean(),data[864:936].mean(),data[936:1008].mean(),
    data[1008:1080].mean(),data[1080:1152].mean(),data[1152:1224].mean(),data[1224:1296].mean(),data[1296:1368].mean(),data[1368:1440].mean(),data[1440:1512].mean()
    ,data[1512:1584].mean(),data[1584:1656].mean(),data[1656:1728].mean(),data[1728:1800].mean(),data[1800:1872].mean(),data[1872:1944].mean(),data[1944:2016].mean(),
    data[2016:2088].mean(),data[2088:2160].mean(),data[2160:2232].mean(),data[2232:2304].mean(),data[2304:2376].mean(),data[2376:2448].mean(),data[2448:2476].mean()]
    dates=["6/10/2020","6/11/2020","6/12/2020","6/13/2020","6/14/2020","6/15/2020","6/16/2020","6/17/2020","6/18/2020","6/19/2020"," 6/20/2020",
    "6/21/2020","6/22/2020","6/23/2020","6/24/2020","6/25/2020","6/26/2020","6/27/2020","6/28/2020","6/29/2020","6/30/2020","7/1/2020","7/2/2020",
    "7/3/2020","7/4/2020","7/5/2020","7/6/2020","7/7/2020","7/8/2020","7/9/2020","7/10/2020","7/11/2020","7/12/2020","7/13/2020","7/14/2020"]    
    d={'Temperature':temperature,'Date':dates}
    df_print=pd.DataFrame(d)
    df_print.sort_values(by=['Temperature','Date'],axis=0,ascending=False)
    
    print (df_print)

plt.show()