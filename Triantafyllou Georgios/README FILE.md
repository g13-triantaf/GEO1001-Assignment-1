Assignment 1-Georgios Triantafyllou

This is a README file to describe how to run geo1001_hw01

The geo1001_hw01 file contains the whole code in order to reach the assignment's requested results. By running the python file all the results appear at once when the run finishes.

In the beginning of the file, some basic libraries were imported, in order to complete the wanted tasks.

The data files were converted to .csv and where imported in python with encoding UTF-8. As a result a dataframe was created for every data file (df_A,df_B etc.). The first 5 rows of each file were skipped, because they did not include numerical values.

# A1

In this part mean, variance annd standard deviation are calculating for each sensor/each dataframe and the results are saved in an excel file. Then the requested figures are creating.

# A2

For this part, no new variables were created and the figures (PMF,PDF,CDF,KDE) can be plotted based on the existing information using the proper functions.

# A3

Crosswind and WBGT data was called in the same way as described in part A1 for the temperature data. Pearson and Spearman coefficients were calculated for each combination of sensors (10 combinations). Additonally interpolations had to be made for variables with different length. The coefficient results are exported in a csv file named "Correlation". Finally 3 scatter plots were created, one for each variable.

# A4

In order to calculate the confidence intervals for temperatures and wind speed a function was created which takes two variables, the data and the confidence level which is 0.95. The function uses the scipy.stats library to calculate the 3 intervals. Then, to calculate the intervals for each variable, the function is called for each data and for confidence level 0.95. The results are saved in a csv file named "Confidence_intervals". Regarding the t testing and p values, the statistics library was used for each variable and The results are visible after running the code.

#Bonus question

A function named average_temperature was created that takes data as its only variable. The function calculates the maximum and minimum temperatures in order to find which day is the hottest and coolest respectively, during the days that the data was acquired.