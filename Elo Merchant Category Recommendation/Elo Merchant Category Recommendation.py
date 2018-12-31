# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# Import needed datasets
train = pd.read_csv('train.csv', parse_dates=['first_active_month'])
test = pd.read_csv('test.csv', parse_dates=['first_active_month'])
historical_transactions = pd.read_csv('historical_transactions.csv', parse_dates=['purchase_date'])
new_merchant_transactions = pd.read_csv('new_merchant_transactions.csv', parse_dates=['purchase_date'])
merchants = pd.read_csv('merchants.csv')

# Check features types
for df in [train, test, historical_transactions, new_merchant_transactions, merchants]:
    display(df.info())
    
# Add 'card_id' to 'merchants' dataset to facilitate the merging
joined_dfs = pd.concat([historical_transactions[['card_id', 'merchant_id']], new_merchant_transactions[['card_id', 'merchant_id']]])
merchants = pd.merge(joined_dfs, merchants, on='merchant_id')

# Merge all datasets with 'train' to check the how the different features are reacting with 'target'
hist_df = pd.merge(train, historical_transactions, on='card_id', how='left')
new_df = pd.merge(train, new_merchant_transactions, on='card_id', how='left')
mer_df = pd.merge(train, merchants, on='card_id', how='left')

# Set the sizes of different graphs
plt.rcParams['figure.figsize'] = 16, 4

# 'target' feture histogram
plt.rcParams['figure.figsize'] = 16, 7
a = sns.distplot(train['target'])

# 'first_active_month' histogram (inspired by Sudalairaj Kumar's kernel: "https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-elo")
dates = train['first_active_month'].dt.date.value_counts()
dates = dates.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(dates.index, dates.values, color='green')
plt.xticks(rotation='vertical')
plt.show()


# Create a new dataset
df = hist_df

# Object that holds discrete features in the above dataset
discrete = ['feature_1', 'feature_2', 'feature_3', 'category_1', 'category_2', 'category_3', 'authorized_flag',
          'installments', 'month_lag']

# Object that holds continuous features in the above dataset
continuous = ['city_id', 'merchant_category_id', 'purchase_amount', 'subsector_id']

# Loop through discrete features and use violin plot to visualize each feature
for column in discrete:
    a1 = sns.violinplot(df[column], df['target'])
    plt.show()

# Loop through continuous features and use violin plot to visualize each feature
for column in continuous:
    j = sns.jointplot(data=df, x = column, y='target')
    j.fig.set_figwidth(16)
    j.fig.set_figheight(5)
    
    
# Create a new dataset
df = new_df

# Object that holds discrete features in the above dataset
discrete = ['feature_1', 'feature_2', 'feature_3', 'category_1', 'category_2', 'category_3', 'authorized_flag',
          'installments', 'month_lag']

# Object that holds continuous features in the above dataset
continuous = ['city_id', 'merchant_category_id', 'purchase_amount', 'subsector_id']

# Loop through discrete features and use violin plot to visualize each feature
for column in discrete:
    a1 = sns.violinplot(df[column], df['target'])
    plt.show()

# Loop through continuous features and use violin plot to visualize each feature
for column in continuous:
    j = sns.jointplot(data=df, x = column, y='target')
    j.fig.set_figwidth(16)
    j.fig.set_figheight(5)
    
    
# Create a new dataset
df = mer_df

# Object that holds discrete features in the above dataset
discrete = ['category_1', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']

# Loop through discrete features and use violin plot to visualize each feature
for column in discrete:
    a1 = sns.violinplot(df[column], df['target'])
    plt.show()
    
    
# Create a list that holds all dataset
dfs = [train, test, historical_transactions, new_merchant_transactions, merchants]

# Create a title list with the name of each dataset 
dfs_titles = ['train', 'test', 'historical_transactions', 'new_merchant_transactions', 'merchants']
counter = 0

# Loop on each dataset to check the missing values within it
for df in dfs:
    print('\nDisplay the number of missing observations within "%s" dataset' % dfs_titles[counter])
    display(df.isnull().sum().reset_index(name='count').sort_values(by='count', ascending=False))
    counter += 1


# Calculate the time difference between the reference date (2018, 2, 1) and first active month feature (Peter Hurford's kernel: "https://www.kaggle.com/peterhurford/you-re-going-to-want-more-categories-lb-3-737")
train['elapsed_time'] = (datetime.date(2018, 2, 1) - train['first_active_month'].dt.date).dt.days
test['elapsed_time'] = (datetime.date(2018, 2, 1) - test['first_active_month'].dt.date).dt.days

# Break down the date feature & delete the 'first_active_month' feature (Chau Huynh's kernel: "https://www.kaggle.com/chauhuynh/my-first-kernel-3-699")
for df in [train, test]:
    df["first_year"] = df["first_active_month"].dt.year
    df["first_month"] = df["first_active_month"].dt.month
    df['weekofyear'] = df['first_active_month'].dt.weekofyear
    df['dayofweek'] = df['first_active_month'].dt.dayofweek
    df['weekend'] = (df.first_active_month.dt.weekday >=5).astype(int)
    df['first_year'] = df['first_year'].fillna(df['first_year'].median())
    df['first_month'] = df['first_month'].fillna(df['first_month'].median())
    df.drop(['first_active_month'], axis=1, inplace=True)

# Omitting observations with missing 'merchant_id'
historical_transactions = historical_transactions[historical_transactions['merchant_id'].notnull()]
new_merchant_transactions = new_merchant_transactions[new_merchant_transactions['merchant_id'].notnull()]

# Convert categorical features to quantitative
for df in [historical_transactions, new_merchant_transactions]:
    for column in ['authorized_flag', 'category_1']:
        df[column] = df[column].map({'Y':1, 'N':0})

# Break down the date feature & delete the unneeded features
for df in [historical_transactions, new_merchant_transactions]:
    df['purchase_year'] = df["purchase_date"].dt.year
    df["purchase_month"] = df["purchase_date"].dt.month
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['dayofweek'] = df['purchase_date'].dt.dayofweek
    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)
    df['hour'] = df['purchase_date'].dt.hour
    df.drop(['purchase_date', 'category_2', 'category_3'], axis=1, inplace=True)

# Convert categorical features to quantitative
for df in [merchants]:
    for column in ['category_1', 'category_4']:
        df[column] = df[column].map({'Y':1, 'N':0})
    for column in ['most_recent_sales_range', 'most_recent_purchases_range']:
        df[column] = df[column].map({'A':1, 'B':0, 'C':3, 'D':4, 'E':5})
    
# Replace missing values with median
for column in ['avg_sales_lag3', 'avg_sales_lag6', 'avg_sales_lag12']:
        merchants[column] = merchants[column].fillna(merchants[column].median())
        
# Delete 'category_2' feature
merchants.drop(['category_2'], axis=1, inplace=True)


# Aggregate all 'historical_transactions' dataset's features
agg_func = {}
for column in historical_transactions.columns:
    if column not in  ('card_id', 'merchant_id'):
        agg_func[column] = ['min', 'max', 'sum', 'nunique', 'count', 'mean', 'median', 'std']
    if column in  ('card_id'):
        agg_func[column] = ['size']
historical_transactions = historical_transactions.groupby(by='card_id').agg(agg_func)

# Aggregate all 'new_merchant_transactions' dataset's features
agg_func = {}
for column in new_merchant_transactions.columns:
    if column not in  ('card_id', 'merchant_id'):
        agg_func[column] = ['min', 'max', 'sum', 'nunique', 'count', 'mean', 'median', 'std']
    if column in  ('card_id'):
        agg_func[column] = ['size']
new_merchant_transactions = new_merchant_transactions.groupby(by='card_id').agg(agg_func)

# Aggregate all 'merchants' dataset's features
agg_func = {}
for column in merchants.columns:
    if column not in ('card_id', 'merchant_id'):
        agg_func[column] = ['min', 'max', 'sum', 'nunique', 'count', 'mean', 'median', 'std']
    if column in  ('card_id'):
        agg_func[column] = ['size']
    if column in ('merchant_id'):
        agg_func[column] = ['count']
merchants = merchants.groupby(by='card_id').agg(agg_func)

# Convert columns from two levels (after the aggregation, features were displayed as two levels) to one level
for df in [historical_transactions, new_merchant_transactions, merchants]:
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df['card_id'] = df.index
    
# Distinguish datasets' features with different features' names with an indication to which feature belongs to which dataset
counter = 0
abr = ['his', 'new', 'mer']
for df in [historical_transactions, new_merchant_transactions, merchants]:
    df.columns = abr[counter] + '_' + df.columns
    df.rename(columns={df.columns[-1]: 'card_id'}, inplace=True)
    counter += 1
    
    
# Merge different datasets with 'train'
new_train = pd.merge(train, historical_transactions, on='card_id')
new_train = pd.merge(new_train, new_merchant_transactions, on='card_id', how='left')
new_train = pd.merge(new_train, merchants, on='card_id')
new_train = new_train.set_index('card_id')

# Merge different datasets with 'test'
new_test = pd.merge(test, historical_transactions, on='card_id')
new_test = pd.merge(new_test, new_merchant_transactions, on='card_id', how='left')
new_test = pd.merge(new_test, merchants, on='card_id')
new_test = new_test.set_index('card_id')

# Create a correlation dataframe between 'target' and all features
corr = abs(new_train.corr().target).sort_values(ascending=False).reset_index()

# Create a new_train dataset with all features that has a correlation with 'target' and neglect those with 'NAN'
new_train = new_train[corr[corr['target'].notna()]['index']]

# Create a new_test dataset with all features that has a correlation with 'target' and neglect those with 'NAN'
new_test = new_test[corr[corr['target'].notna()]['index'][1:]]


# Create a copy of new_train dataset
df = new_train

# Create a list with features names that will be displayed
columns = ['new_purchase_year_median', 'mer_most_recent_sales_range_median']

# Loop over the features and create a violin plot for each
for column in columns:
    a1 = sns.violinplot(df[column], df['target'])
    plt.show()
    
# Create a function to calculate the rmse
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Create X and y objects
X = new_train.iloc[:, 1:].values
y = new_train['target'].values

algo = XGBRegressor(n_estimators=150, reg_alpha=1, n_jobs=-1)
accuracies = cross_val_score(estimator = algo, X = X, y = y, cv = 10, n_jobs = -1, scoring = make_scorer(rmse))
print('- rmse using 10 folds is: %.5f' % accuracies.mean())

# Fit the algorithm
classifier = XGBRegressor(n_estimators=150, reg_alpha=1, n_jobs=-1)
classifier.fit(X, y)

# Predicting the test data
test_dataset = new_test.values

# Predict the survival of test dataset
prediction = {'card_id': new_test.index.values, 'target': classifier.predict(test_dataset)}

# Creating prediction file
submission_file = pd.DataFrame(prediction)
submission_file.to_csv('loyalty_score.csv', index=False)