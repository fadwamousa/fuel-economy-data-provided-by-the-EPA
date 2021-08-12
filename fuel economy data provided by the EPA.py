#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# ## Assessing
# 

# 1.number of samples in each dataset
# 2.number of columns in each dataset
# 3.duplicate rows in each dataset
# 4.datatypes of columns
# 5.features with missing values
# 6.number of non-null unique values for features in each dataset
# 7.what those unique values are and counts for each
# 

# In[4]:


df_08 = pd.read_csv('all_alpha_08.csv')
df_08.head(2)
df_08.shape[0]


# In[5]:


df_18 = pd.read_csv('all_alpha_18.csv')
df_18.head(2)
df_18.shape[0]


# In[6]:


df_08.shape[1]


# In[7]:


df_18.shape[1]


# In[10]:


sum(df_08.duplicated()) #number of duplicated rows


# In[11]:


sum(df_18.duplicated()) #number of duplicated rows


# In[12]:


df_08.dtypes


# In[13]:


df_18.dtypes


# In[14]:


df_08.isnull().any()


# In[15]:


df_08.isnull().sum()


# In[16]:


df_18.isnull().sum()


# In[18]:


df_08.nunique() #== len(df_08)


# In[20]:


df_18.nunique()


# ## Cleaning Column Labels

# ### Drop Extraneous Columns

# In[23]:


df_08.columns
df_08.shape[1]


# In[24]:


df_08_columns =['Stnd', 'Underhood ID', 'FE Calc Appr', 'Unadj Cmb MPG'] 
df_08.drop(columns = df_08_columns , inplace = True)


# In[25]:


df_08.columns
df_08.shape[1]


# In[26]:


df_18_columns = ['Stnd', 'Stnd Description', 'Underhood ID', 'Comb CO2']
df_18.drop(columns = df_18_columns , inplace = True)


# In[27]:


df_18.shape[1]


# ### Rename columns

# Change the "Sales Area" column label in the 2008 dataset to "Cert Region" for consistency

# In[29]:


df_08.rename(columns = {'Sales Area':'Cert Region'} , inplace = True)


# In[33]:


df_08.columns


# Rename all column labels to replace spaces with underscores and convert everything to lowercase

# In[36]:


df_08.rename(columns= lambda x: x.strip().lower().replace(" ","_") , inplace = True)
df_08.head(1)


# In[37]:


df_18.rename(columns = lambda x : x.strip().lower().replace(" ","_"),inplace = True)
df_18.head(1)


# In[38]:


df_08.columns == df_18.columns


# In[39]:


(df_08.columns == df_18.columns).all()


# ## save new datasets for next section
# 

# In[40]:


df_08.to_csv('data_08_v1.csv', index=False)
df_18.to_csv('data_18_v1.csv', index=False)


# In[41]:


# load datasets
import pandas as pd
df_08 = pd.read_csv('data_08_v1.csv')
df_18 = pd.read_csv('data_18_v1.csv')


# In[43]:


df_08.head(1)


# In[57]:


# filter datasets for rows following California standards
df_08 = df_08[df_08['cert_region'] == 'CA']
df_18 = df_18[df_18['cert_region'] == 'CA']


# In[54]:


# confirm only certification region is California
df_08['cert_region'].unique()


# In[58]:


# confirm only certification region is California
df_18['cert_region'].unique()


# In[59]:


df_08.drop('cert_region' , axis = 1 , inplace = True)
df_18.drop('cert_region' , axis = 1 , inplace = True)


# ## Drop Rows with Missing Values

# In[63]:


df_08.dropna(inplace = True)


# In[65]:


df_18.dropna(inplace = True)


# In[68]:


df_08.isnull().sum()
df_18.isnull().sum()


# ### Dedupe Data

# In[69]:


sum(df_08.duplicated())


# In[70]:


sum(df_18.duplicated())


# In[71]:


df_08.drop_duplicates(inplace = True)


# In[72]:


df_18.drop_duplicates(inplace = True)


# In[73]:


sum(df_18.duplicated())


# In[74]:


sum(df_08.duplicated())


# In[75]:


# save progress for the next section
df_08.to_csv('data_08_v2.csv', index=False)
df_18.to_csv('data_18_v2.csv', index=False)


# In[134]:


import pandas as pd
df_08 = pd.read_csv('data_08_v2.csv')
df_18 = pd.read_csv('data_18_v2.csv')
df_08.shape


# In[81]:


df_08['cyl'].value_counts()


# In[82]:


df_18['cyl'].dtypes
df_08['cyl'].dtypes


# In[94]:


df_18['cyl'].dtypes


# ### convert cyl form both dataset into int data type instead float or string

# In[83]:


df_08['air_pollution_score'].dtypes


# In[84]:


df_08['air_pollution_score'].value_counts()


# In[87]:


df_18['air_pollution_score'].dtypes


# In[88]:


df_18['air_pollution_score'].value_counts()


# ### convert air_pollution_score form both dataset into float data type instead int or string

# In[89]:


df_08['greenhouse_gas_score'].dtypes


# In[90]:


df_18['greenhouse_gas_score'].dtypes


# In[91]:


# load datasets
import pandas as pd
df_08 = pd.read_csv('data_08_v2.csv')
df_18 = pd.read_csv('data_18_v2.csv')


# In[92]:


df_08['cyl'] = df_08['cyl'].str.extract('(\d+)').astype(int)


# In[93]:


df_08['cyl'].dtypes


# In[97]:


df_18['cyl'] = df_18['cyl'].astype(int)


# In[98]:


df_18['cyl'].dtypes


# In[99]:


df_08.to_csv('data_08_v3.csv', index=False)
df_18.to_csv('data_18_v3.csv', index=False)


# In[100]:


# load datasets
import pandas as pd
df_08 = pd.read_csv('data_08_v3.csv')
df_18 = pd.read_csv('data_18_v3.csv')


# In[101]:


df_08['air_pollution_score'].dtypes


# In[102]:


df_08['air_pollution_score'] = df_08['air_pollution_score'].str.extract('(\d+)').astype(float)


# In[103]:


df_08['air_pollution_score'].dtypes


# In[104]:


df_18['air_pollution_score'].dtypes


# In[105]:


df_18['air_pollution_score'] = df_18['air_pollution_score'].astype(float)


# In[106]:


df_18['air_pollution_score'].dtypes


# In[107]:


df_08['air_pollution_score'].value_counts()


# ### convert the fields with two values into two rows and apply the function on that fields

# In[174]:


hb_08=df_08[df_08['fuel'].str.contains('/')]
hb_18=df_18[df_18['fuel'].str.contains('/')] #الحقل الواحد بيختوي علي اكثر من قيمة


# In[115]:


df1 = hb_08.copy()  # data on first fuel type of each hybrid vehicle
df2 = hb_08.copy()  # data on second fuel type of each hybrid vehicle

# Each one should look like this
df1


# In[ ]:


# columns to split by "/"
split_columns = ['fuel', 'air_pollution_score', 'city_mpg', 'hwy_mpg', 'cmb_mpg', 'greenhouse_gas_score']

# apply split function to each column of each dataframe copy
for c in split_columns:
    df1[c] = df1[c].apply(lambda x: x.split("/")[0])
    df2[c] = df2[c].apply(lambda x: x.split("/")[1])


# In[120]:


df1


# In[121]:


df2


# In[122]:


# combine dataframes to add to the original dataframe
new_rows = df1.append(df2)

# now we have separate rows for each fuel type of each vehicle!
new_rows


# In[131]:


# drop the original hybrid rows
df_08.drop(hb_08.index, inplace=True)

# add in our newly separated rows
df_08 = df_08.append(new_rows, ignore_index=True)
df_08


# In[132]:


# check that all the original hybrid rows with "/"s are gone
df_08[df_08['fuel'].str.contains('/')]


# In[161]:


df_08.shape
df_18.shape


# # Repeat this process for the 2018 dataset

# In[138]:


hb_18=df_18[df_18['fuel'].str.contains('/')] #الحقل الواحد بيختوي علي اكثر من قيمة
df1 = hb_18.copy()
df2 = hb_18.copy()


# In[140]:


df1.shape


# In[ ]:


# columns to split by "/"
split_columns = ['fuel', 'city_mpg', 'hwy_mpg', 'cmb_mpg']

# apply split function to each column of each dataframe copy
for c in split_columns:
    df1[c] = df1[c].apply(lambda x: x.split("/")[0])
    df2[c] = df2[c].apply(lambda x: x.split("/")[1])
    


# In[160]:


df2.shape


# In[159]:


# append the two dataframes
new_rows = df1.append(df2)

# drop each hybrid row from the original 2018 dataframe
# do this by using Pandas drop function with hb_18's index
df_18.drop(hb_18.index, inplace=True)

# append new_rows to df_18
df_18 = df_18.append(new_rows, ignore_index=True)
df_18.shape


# In[162]:


df_18[df_18['fuel'].str.contains('/')]


# In[163]:


df_18.shape


# In[165]:


df_18['air_pollution_score'].dtypes


# In[167]:


df_18['air_pollution_score'] = df_18['air_pollution_score'].astype(float)


# In[168]:


df_08['air_pollution_score'].dtypes


# In[170]:


df_08['air_pollution_score'] = df_08['air_pollution_score'].str.extract('(\d+)').astype(float)


# In[171]:


df_08['air_pollution_score'].dtypes


# In[173]:


df_08.to_csv('data_08_v4.csv', index=False)
df_18.to_csv('data_18_v4.csv', index=False)


# In[175]:


# load datasets
import pandas as pd
df_08 = pd.read_csv('data_08_v4.csv')
df_18 = pd.read_csv('data_18_v4.csv')


# In[177]:


df_08.greenhouse_gas_score.dtypes


# In[ ]:


### convert this column into int from string
df_08['greenhouse_gas_score'] = df_08['greenhouse_gas_score'].str.extract('(\d+)').astype(int)


# In[190]:


df_08.city_mpg.dtypes


# In[189]:


columns = ['city_mpg','hwy_mpg','cmb_mpg']
for c in columns:
    df_08[c] = df_18[c].str.extract('(\d+)').astype(float)
    df_18[c] = df_18[c].str.extract('(\d+)').astype(float)
    


# ## Fix greenhouse_gas_score datatype
# 

# In[ ]:



### convert this column into int from string
df_08['greenhouse_gas_score'] = df_08['greenhouse_gas_score'].str.extract('(\d+)').astype(int)
df_08['greenhouse_gas_score'].dtype


# In[193]:


df_08.dtypes


# In[194]:


df_18.dtypes


# In[195]:


df_08.columns == df_18.columns


# In[196]:


(df_08.columns == df_18.columns).all()


# In[199]:


# Save your final CLEAN datasets as new files!
df_08.to_csv('clean_08.csv', index=False)
df_18.to_csv('clean_18.csv', index=False)


# In[205]:


# load datasets
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df_08 = pd.read_csv('clean_08.csv')
df_18 = pd.read_csv('clean_18.csv')


# ### Exploring with Visuals

# In[206]:


df_08.plot(kind = "hist",figsize=(8,8));


# In[215]:


df_08.hist(figsize=(8,8)); # for numerical columns


# In[211]:


df_08['greenhouse_gas_score'].value_counts().plot(kind='hist',figsize = (2,2));


# In[212]:


df_18['greenhouse_gas_score'].value_counts().plot(kind='hist',figsize = (2,2));


# In[213]:


df_18.hist(figsize=(8,8)); # for numerical columns


# In[216]:


df_08.plot(x = 'displ' , y = 'cmb_mpg' , kind = 'scatter' , figsize=(8,8))


# In[218]:



df_08.plot(x = 'greenhouse_gas_score' , y = 'cmb_mpg' , kind = 'bar' , figsize=(8,8))


# ### Q1: Are more unique models using alternative sources of fuel? By how much?

# In[219]:


df_08.fuel.value_counts()


# In[220]:


df_18.fuel.value_counts()


# In[225]:


df_08[(df_08['fuel'] == 'CNG') & (df_08['fuel'] == 'ethanol')]['model'].nunique()


# In[229]:


# how many unique models used alternative sources of fuel in 2008
alt_08 = df_08.query('fuel in ["CNG", "ethanol"]').model.nunique()
alt_08


# In[227]:


# how many unique models used alternative sources of fuel in 2018
alt_18 = df_18.query('fuel in ["Ethanol", "Electricity"]').model.nunique()
alt_18


# In[230]:


plt.bar(["2008", "2018"], [alt_08, alt_18])
plt.title("Number of Unique Models Using Alternative Fuels")
plt.xlabel("Year")
plt.ylabel("Number of Unique Models");


# In[231]:


# total unique models each year
total_08 = df_08.model.nunique()
total_18 = df_18.model.nunique()
total_08, total_18


# In[232]:


prop_08 = alt_08/total_08
prop_18 = alt_18/total_18
prop_08, prop_18


# In[233]:


plt.bar(["2008", "2018"], [prop_08, prop_18])
plt.title("Proportion of Unique Models Using Alternative Fuels")
plt.xlabel("Year")
plt.ylabel("Proportion of Unique Models");


# ### Q2: How much have vehicle classes improved in fuel economy?  

# In[234]:


veh_08 = df_08.groupby('veh_class').cmb_mpg.mean()
veh_08


# In[235]:


veh_18 = df_18.groupby('veh_class').cmb_mpg.mean()
veh_18


# In[236]:


# how much they've increased by for each vehicle class
inc = veh_18 - veh_08
inc


# In[242]:


# only plot the classes that exist in both years
inc.dropna(inplace=True)
plt.subplots(figsize=(8, 5))
plt.bar(inc.index, inc)
plt.title('Improvements in Fuel Economy from 2008 to 2018 by Vehicle Class')
plt.xlabel('Vehicle Class')
plt.ylabel('Increase in Average Combined MPG');


# ## Merging Datasets

# In[243]:


# load datasets
import pandas as pd
df_08 = pd.read_csv('clean_08.csv')
df_18 = pd.read_csv('clean_18.csv')


# In[244]:


# rename 2008 columns
df_08.rename(columns = lambda x:x[:10]+'_2008' , inplace = True)


# In[245]:


# view to check names
df_08.head()


# In[246]:


df_combined = df_08.merge(df_18,left_on = "model_2008",right_on="model",how="inner")


# In[248]:


# view to check merge
df_combined.head()


# In[249]:


df_combined.to_csv('combined_dataset.csv', index=False)


# ###  Q5: For all of the models that were produced in 2008 that are still being produced now, how much has the mpg improved and which vehicle improved the most?

# #### 1. Create a new dataframe, model_mpg, that contain the mean combined mpg values in 2008 and 2018 for each unique model¶

# In[250]:


# load dataset
import pandas as pd
df = pd.read_csv('combined_dataset.csv')


# In[252]:


df.columns


# In[258]:


model_mpg = df.groupby('model').mean()[['cmb_mpg_2008', 'cmb_mpg']]


# In[259]:


model_mpg


# In[260]:


model_mpg['mpg_change'] = model_mpg['cmb_mpg'] - model_mpg['cmb_mpg_2008']


# In[261]:


model_mpg.head()


# ### 3. Find the vehicle that improved the most
# Find the max mpg change, and then use query or indexing to see what model it is!

# In[262]:


max_change = model_mpg['mpg_change'].max()
max_change


# In[263]:


model_mpg[model_mpg['mpg_change'] == max_change]


# In[ ]:




