#Import the necessary libraries
import pandas as pd
import numpy as np

heart = pd.read_csv("heart.csv")
heart.head(10)
# for male 0,for Female 1
age_enco = {'M':0,'F':1}
heart["Sex"] = heart["Sex"].apply(lambda x: age_enco[x])
# for male 0,for Female 1
chestpaint_enco = {'ASY':0,'NAP':1,'ATA':2,'TA':3}
heart["ChestPainType"] = heart["ChestPainType"].apply(lambda x: chestpaint_enco[x])

# find the categorical columns
cat_col = []
for x in heart.dtypes.index:
    if heart.dtypes[x] == 'object':
        cat_col.append(x)

heart_data = pd.get_dummies(heart,columns=cat_col,drop_first=True)


 #Scaling
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
#Columns that need to be scaled
cols = ["RestingBP","Cholesterol","MaxHR"]
scaled = st.fit_transform(heart_data[cols])
scaled_to_df = pd.DataFrame(scaled,columns=["RestingBP_sc","Cholesterol_sc","MaxHR_sc"])
scaled_to_df.head()
heart_data = pd.concat((scaled_to_df,heart_data),axis=1)
heart_data = heart_data.drop(cols,axis=1)
heart_data.head()

X = heart_data.drop("HeartDisease",axis=1)
y = heart_data["HeartDisease"]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# initialize models
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier


knn = KNeighborsClassifier()
rf = RandomForestClassifier()
ada_bost = AdaBoostClassifier()
grad_bost = GradientBoostingClassifier()
dec_tree = DecisionTreeClassifier()

models = [knn,rf,ada_bost,grad_bost,dec_tree]

for model in models:
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    clas_rep = classification_report(y_test,y_pred)
    print(f"model name: {model}\n classification report:\n{clas_rep}")


# GradientBoostingClasifier gridsearch cross validation
from sklearn.model_selection import GridSearchCV

params = {'learning_rate':[0.1,0.01,0.2,0.5],
         'n_estimators':[100,1000,200,150],
         'min_samples_split':[2,3,5,10],
         'min_samples_leaf':[1,2,5,10]}

grid_cv_model = GridSearchCV(grad_bost,params,cv=10,n_jobs=-1)
grid_cv_model.fit(X_train,y_train)
print(grid_cv_model.best_estimator_)
print(grid_cv_model.best_params_)
print(grid_cv_model.best_score_)


# best model with the best params
final_model = GradientBoostingClassifier(learning_rate=0.1, min_samples_leaf = 2 , min_samples_split = 2 , n_estimators = 100 , loss='squared_error' )
final_model.fit(X_train,y_train)
y_pred = final_model.predict(X_test)
print(classification_report(y_test,y_pred))


import pickle
# Saving model to disk
pickle.dump(final_model, open('model.pkl','wb'))

     # Loading model to compare the results
final_model = pickle.load(open('model.pkl','rb'))
print(final_model.predict([[2,9,6]]))


    