#%% Import libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import pickle

# %%Load the dataset
df = pd.read_csv('insurance.csv')

# %%
df.head()

#%%
df.shape
#%%

df.info()
#%%
df.describe()

#%%

df.age.hist()

#%%
df.bmi.hist()

#%%
df.charges.hist()

#%%

sns.kdeplot(df['charges'])

#%%

sns.boxenplot(df['charges'])
#%%
df.isnull().sum()

#%%


df.children.value_counts()

#%%
df.head()

#%%
sns.boxplot(data=df , x= 'region', y='charges')
# %%
sns.boxplot(data=df , x= 'smoker', y='charges')

#%%
sns.boxplot(data=df , x= 'sex', y='charges')

#%%
sns.scatterplot(data=df , x= 'age', y='charges', hue='sex')

#%%
sns.scatterplot(data=df , x= 'bmi', y='charges')
#%%

# Separate features and target variable
X = df.drop('charges', axis=1)
y = df['charges']


# Log-transform the target variable
y = np.log(y)

# %%Split the data into training and test sets
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.25, random_state=0)

# Define preprocessing pipelines for numeric and categorical features
num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder())
])

preprocessor_pipeline = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, make_column_selector(dtype_exclude=object)),
        ('cat', cat_pipeline, make_column_selector(dtype_include=object))
    ]
)

# Define models to evaluate
models = [
    ('LinearRegression', LinearRegression()),
    ('Lasso', Lasso()),
    ('RandomForestRegressor', RandomForestRegressor()),
    ('KNN', KNeighborsRegressor())
]

# Cross-validation results dictionary
cv_results = {}

# Define cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Evaluate each model
for name, model in models:
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_pipeline),
        ('model', model)
    ])
    cv_result = cross_val_score(model_pipeline, trainX, trainY, cv=kf, scoring='r2')
    cv_results[name] = np.mean(cv_result)
    print(f"Model: {name}, Mean R2 Score: {cv_results[name]}")

#%% Fit and evaluate the best model on the test set (optional)
best_model_name = max(cv_results, key=cv_results.get)
best_model = dict(models)[best_model_name]
best_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_pipeline),
    ('model', best_model)
])
best_model_pipeline.fit(trainX, trainY)
pred = best_model_pipeline.predict(testX)
test_score = r2_score(testY, pred)
print(f"Best Model: {best_model_name}, Test R2 Score: {test_score}")



# %% Hyperparameter grid for RandomForestRegressor
rf_param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20, 30]
}


# %%
leg_model = Pipeline([
    ('preprocessor', preprocessor_pipeline),
    ('model',RandomForestRegressor())
])

gridsearch = GridSearchCV(leg_model, param_grid=rf_param_grid,cv=5,scoring='r2')
gridsearch.fit(trainX,trainY)
gridsearch.best_params_
#%%
model = gridsearch.best_estimator_
pred = model.predict(testX)
score = r2_score(testY,pred)
print(score)
# %%
pickle.dump(model,open('model.pkl','wb'))
# %%
model = pickle.load(open('model.pkl', 'rb'))
# %%
model.predict(testX)
# %%