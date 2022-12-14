#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('insurance.csv')


# ### Displaying top 5 rows

# In[3]:


df.head()


# In[4]:


df.tail()


# ### Finding No. of rows & Columns

# In[5]:


df.shape


# ### Getting Information about Dataset

# In[6]:


df.info()


# ### Checking null value in Dataset

# In[7]:


df.isnull()


# In[8]:


df.isnull().sum() # sum of null values in each column


# ### Getting Overall Statistics of Dataset

# In[9]:


df.describe()  


# In[10]:


df.describe(include='all') # statistics for caterogical and numerical column both


# In[11]:


df.head()


# ### Converting Columns from string to Integer

# In[12]:


df['sex'].unique()


# In[13]:


df['sex']= df['sex'].map({'female':0,'male':1})


# In[14]:


df['sex']


# In[15]:


df.head()


# In[16]:


df['smoker'].unique()


# In[17]:


df['smoker']= df['smoker'].map({'yes':1,'no':0})


# In[18]:


df['smoker']


# In[19]:


df['region'].unique()


# In[20]:


df['region']=df['region'].map({'southwest':1,'southeast':2,'northwest':3,'northeast':4})


# In[21]:


df['region']


# In[22]:


df.head()


# In[23]:


df.columns


# ### Storing Features Matrix in X and response in Y

# In[24]:


X = df.drop(['charges'],axis=1)


# In[25]:


X


# In[26]:


Y= df['charges']


# In[27]:


Y


# ### Importing the Models

# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)


# In[30]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# ### Model Trainning

# In[31]:


lr=LinearRegression()
lr.fit(X_train,Y_train)
sv=SVR()
sv.fit(X_train,Y_train)
rf= RandomForestRegressor()
rf.fit(X_train,Y_train)
gb=GradientBoostingRegressor()
gb.fit(X_train,Y_train)


# ### Prediction on Test Data

# In[32]:


Y_pred1 = lr.predict(X_test)
Y_pred2 = sv.predict(X_test)
Y_pred3 = rf.predict(X_test)
Y_pred4 = gb.predict(X_test)

df1 = pd.DataFrame({'Actual':Y_test,'lr':Y_pred1,'sv':Y_pred2,'rf':Y_pred3,'gb':Y_pred4 } )


# In[33]:


df1


# ### Comparing performance Visually

# In[34]:


import matplotlib.pyplot as plt


# In[35]:


plt.subplot(221)
plt.plot(df1['Actual'].iloc[0:15],label='Actual')
plt.plot(df1['lr'].iloc[0:15],label='lr')
plt.legend()


# In[36]:


plt.subplot(221)
plt.plot(df1['Actual'].iloc[0:15],label='Actual')
plt.plot(df1['lr'].iloc[0:15],label='lr')
plt.legend()

plt.subplot(222)
plt.plot(df1['Actual'].iloc[0:15],label='Actual')
plt.plot(df1['rf'].iloc[0:15],label='rf')
plt.legend()

plt.subplot(223)
plt.plot(df1['Actual'].iloc[0:15],label='Actual')
plt.plot(df1['sv'].iloc[0:15],label='sv')
plt.legend()

plt.subplot(224)
plt.plot(df1['Actual'].iloc[0:15],label='Actual')
plt.plot(df1['gb'].iloc[0:15],label='gb')
plt.legend()


# ### Evaluating the Models

# In[38]:


from sklearn import metrics


# In[39]:


score1=metrics.r2_score(Y_test,Y_pred1)
score2=metrics.r2_score(Y_test,Y_pred2)
score3=metrics.r2_score(Y_test,Y_pred3)
score4=metrics.r2_score(Y_test,Y_pred4)


# In[40]:


print(score1,score2,score3,score4)


# In[45]:


s1=metrics.mean_absolute_error(Y_test,Y_pred1)
s2=metrics.mean_absolute_error(Y_test,Y_pred2)
s3=metrics.mean_absolute_error(Y_test,Y_pred3)
s4=metrics.mean_absolute_error(Y_test,Y_pred4)


# In[46]:


print(s1,s2,s3,s4)


# ### Charges prediction for new customer

# In[48]:


data ={'age':40,
       'sex':1,
       'bmi':50.5,
       'children':3,
       'smoker':1,
       'region':2
        }
df = pd.DataFrame(data, index=[0])
df


# In[49]:


new_pred =gb.predict(df)


# In[50]:


print(new_pred)


# In[51]:


gb= GradientBoostingRegressor()
gb.fit(X,Y)


# ### Saving the Model

# In[52]:


import joblib
joblib.dump(gb,'model_joblib_gb')


# In[53]:


model=joblib.load('model_joblib_gb')
model.predict(df)


# ### GUI

# In[55]:


from tkinter import*


# In[56]:


import joblib


# In[ ]:


def show_entry():
    p1 = float(e1.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())
    
    model = joblib.load('model_joblib_gb')
    result=model.predict([[p1,p2,p3,p4,p5,p6]])
    Label(master, text = "Insurance Cost").grid(row=8)
    Label(master,text=result).grid(row=8,column=1)

master =Tk()
master.title("Insurance Cost Prediction")
label = Label(master,text ="Insurance Cost Prediction",bg = "blue",fg ="white").grid(row=0,columnspan=2)
Label(master,text = "Enter Your Age").grid(row=2)
Label(master,text = "Male or Female [ 1/0]").grid(row=3)
Label(master,text = "Enter Your BMI Value").grid(row=4)
Label(master,text = "Enter Number of children").grid(row=5)
Label(master,text = "Smoker Yes/No [1/0]").grid(row=6)
Label(master,text = "Region [1-4]").grid(row=7)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)

e1.grid(row=2,column=1)
e2.grid(row=3,column=1)
e3.grid(row=4,column=1)
e4.grid(row=5,column=1)
e5.grid(row=6,column=1)
e6.grid(row=7,column=1)

Button(master,text="predict",command=show_entry).grid()

mainloop()


# In[ ]:




