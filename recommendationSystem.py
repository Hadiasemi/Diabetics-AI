#import external libraries and functions
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors

#load dataset from CSV file and show first 5 records
food_bank = pd.read_csv('test_food.csv')
food_bank.columns = ['Shrt_Desc', 'Energ_Kcal', 'Protein_(g)', 'Lipid_Tot_(g)', 'Carbohydrt_(g)', 'Fiber_TD_(g)', 'Sugar_Tot_(g)', 'Sodium_(mg)']
food_bank.head()

# This test farmer will be malnurished 
# plenty of fruit and vegetables
# plenty of bread, rice, potatoes, pasta and other starchy foods
# some milk and dairy foods
# some meat, fish, eggs, beans and other non dairy sources of protein
test_farmer = [ 700, 100, 80]

#Extract only the relevant column from your dataset to reduce computation time.
X = food_bank.iloc[:,[ 1, 2, 3]].values

#Use fit method to create model
nbrs = NearestNeighbors(n_neighbors=1).fit(X)

#Check the recommendation by your model.
print(food_bank.iloc[nbrs.kneighbors([test_farmer])[1][0][0]])
