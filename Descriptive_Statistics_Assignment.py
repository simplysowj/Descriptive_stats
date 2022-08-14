#!/usr/bin/env python
# coding: utf-8

# # Descriptive Statistics and Python Implementation

# ### Statistics

# Statistics is a branch of applied mathematics that involves the collection, description, analysis, and inference of conclusions from quantitative data.

# Statistics is devided in two types:
#     descriptive statistics and Inferential statistics.

# ![stats1.png](attachment:stats1.png)

# ### Descriptive Statistics

# Descriptive statistics provides summaries about either the population data or the sample data. Apart from descriptive statistics, inferential statistics is another crucial branch of statistics that is used to make inferences about the population data.
# 
# Descriptive statistics can be broadly classified into two categories - measures of central tendency and measures of dispersion. 

# Measure of central tendacy:
#     mean
#     mode
#     median
#     
#     
# Measure of dispersion:
#     Range
#     Standard Deviation
#     Variance

# ![stats2.png](attachment:stats2.png)

# ### Measure of central tendacy: 

# ![stats3.png](attachment:stats3.png)

# ### MEAN

# The mean or Average is the Sum of all the values in the dataset divided by the total number of datapoints.
# For example, the sum of the following data set is 20: (2, 3, 4, 5, 6). The mean is 4 (20/5).
#     

# ![stats4.png](attachment:stats4.png)

# ![st6.png](attachment:st6.png)

# #### Calculating Mean for the given dataset(Python implementation)

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("data_ds.csv")


# In[3]:


df


# In[4]:


df['Mthly_HH_Income'].mean()


# Now we will apply the mean formula to compare mean with pandas mean.

# In[5]:


count=0
for i in (df['Mthly_HH_Income']):
    count=count+i
mean=count/len(df['Mthly_HH_Income'])
print(mean)


# As we can see both the answers are correct.

# ### MODE

# The mode is the most Frequently occuring data in the dataset
# 
# For example, in the following list of numbers, 16 is the mode since it appears more times in the set than any other number:
# 
# 3, 3, 6, 9, 16, 16, 16, 27, 27, 37, 48

# ![mode.png](attachment:mode.png)

# In[18]:


df.mode()


# ### MEDIAN

# Median is the value that divides the data into 2 equal parts i.e. number of terms on the right side of it is the same as a number of terms on the left side of it when data is arranged in either ascending or descending order.

# Median will be a middle term if the number of terms is odd
# 
# Median will be the average of the middle 2 terms if a number of terms is even.

# ![median.png](attachment:median.png)

# In[8]:


df["Annual_HH_Income"].median()


# #### Outliers

# In statistics, an outlier is a data point that differs significantly from other observations

# ![outliers.png](attachment:outliers.png)

# Mean is typically the best measure of central tendency because it takes all values into account. But it is easily affected by any extreme value/outlier

# The median is not affected by very large or very small values.

# ### Measure of dispersion

# ![mv.png](attachment:mv.png)

# ### Range

# The range describes the difference between the largest and smallest data point in our data set. The bigger the range, the more is the spread of data and vice versa.

# In[ ]:





# ### Variance

# variance is the average squared distance of the Data points from its mean

# ![var.png](attachment:var.png)

# #### Pandas Implementation

# In[9]:


df['Emi_or_Rent_Amt'].var()


# In[13]:


n = len(df["Emi_or_Rent_Amt"])
mean = sum(df["Emi_or_Rent_Amt"])/n
deviations = [(x - mean) ** 2 for x in df["Emi_or_Rent_Amt"]]
var1=sum(deviations)/n-1
print(var1)


# ### Standard Deviation

# The standard deviation is the square root of variance.

# ![sd.png](attachment:sd.png)

# #### Pandas Implementation

# In[19]:


n = len(df["Emi_or_Rent_Amt"])
mean = sum(df["Emi_or_Rent_Amt"])/n
deviations = [(x - mean) ** 2 for x in df["Emi_or_Rent_Amt"]]
std = (sum(deviations)/n)**0.5
print(std)


# ### Correlation

# Correlation gives a measure of the degree of relationship between two variables.

# In[14]:


df.corr()


# ![corr.png](attachment:corr.png)

# In[23]:


sns.heatmap(df.corr(),annot=True)


# In[25]:


sns.regplot(df['Mthly_HH_Expense'],df['Mthly_HH_Income'])


# ### Normal Distribution

# Normal distribution is also known as the Gaussian distribution, is a probability distribution that is symmetric about the mean, showing that data near the mean are more frequent in occurrence than data far from the mean. In graph form, normal distribution will appear as a bell curve.

# ![nd.png](attachment:nd.png)

# ![nd1.png](attachment:nd1.png)

# ### Features of Normal Distribution

# In normal distributation The mean, mode and median are all equal.
# 
# The curve is symmetric at the center (i.e. around the mean, μ).
# 
# Exactly half of the values are to the left of center and exactly half the values are to the right.
# 
# The total area under the curve is 1.
# 
# 

# ### Positively Skewed & Negatively Skewed Normal Distribution

# In probability theory and statistics, skewness is a measure of the asymmetry of the probability distribution of a random variable. The skewness value can be Positive, zero, Negative.

# Positive Skewness means when the tail on the right side of the distribution is longer.

# Negative Skewness is when the tail of the left side of the distribution is longer.

# The skewness for a normal distribution is zero, and any symmetric data should have a skewness near zero.

# ### Effect on Mean, Median and Mode due to Skewness

# The mean and median will be greater than the mode for positive skewness
# The mean and median will be lesser than the mode for negative skewness
# when the skewness is 0, it means it is a normal distribution and hence, mean = median = mode.

# ![skew.png](attachment:skew.png)

# In[16]:


import seaborn as sns


# In[17]:


sns.distplot(df["Annual_HH_Income"],hist=True,kde=True)


# ### Explain QQ Plot and show the implementation of the same

# A QQ plot; also called a Quantile Quantile plot; is a scatter plot that compares two sets of data. A common use of QQ plots is checking the normality of data. This is considered a normal qq plot, and resembles a standard normal distribution through the reference line and value distribution.

# In[26]:


from scipy import stats
import matplotlib.pyplot as plt

stats.probplot(df['Annual_HH_Income'], dist="norm", plot=plt)
plt.show()


# #### BOX COX

# A Box cox transformation is defined as a way to transform non-normal dependent variables in our data to a normal shape through which we can run a lot more tests than we could have.

# In[29]:


sns.distplot(df['Mthly_HH_Expense'], hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2},
            label = "Non-Normal", color ="green")


# In[ ]:


fitted_data, fitted_lambda = stats.boxcox(original_data)


# In[47]:


fitted_data,fitted_lambda=stats.boxcox(df['Mthly_HH_Expense'])


# In[48]:


fitted_data


# In[51]:



print(f"Lambda value used for Transformation: {fitted_lambda}")


# In[52]:


sns.distplot(fitted_data, hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2},
            label = "Non-Normal", color ="green")


# We can see that the non-normal distribution was converted into a normal distribution or rather close to normal using the SciPy.stats.boxcox().

# In[ ]:




