import pandas as pd
df=pd.read_csv("alzheimer.csv")
df.dropna(inplace=True)
df['Group'].replace(['Nondemented','Demented'],[0,1],inplace=True)
df['M/F'].replace(['M', 'F'],[0, 1], inplace=True)
df=df.drop(df[df["Group"]=="Converted"].index)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

cols=list(df.columns)
x_train,x_test,y_train,y_test = train_test_split(df[cols[1:]],df[cols[0:1]], train_size=0.8, test_size=0.2, shuffle=False)

y_train=y_train.astype('int')
y_test=y_test.astype('int')

y_train=list(y_train["Group"])
y_test=list(y_test["Group"])

model=RandomForestClassifier()
model.fit(x_train,y_train)

feature_imp = pd.Series(model.feature_importances_,index=cols[1:]).sort_values(ascending=False)

cols=list(df.columns)
x_train,x_test,y_train,y_test = train_test_split(df[["MMSE","CDR"]],df[cols[0:1]], train_size=0.8, test_size=0.2, shuffle=False)

y_train=y_train.astype('int')
y_test=y_test.astype('int')

y_train=list(y_train["Group"])
y_test=list(y_test["Group"])

model=RandomForestClassifier()
model.fit(x_train,y_train)

import pickle
pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))







