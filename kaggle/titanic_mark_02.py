# - Predict fate (live/die) of passengers on the Titanic
# Titanic tutorial submission achieves 77.5% accuracy
# I've gotten to 88.9% accuracy
# - Disclaimers:
# I'm only using the training dataset, not the entire dataset (training+test)
# I'm not sure if my prediction code excludes the Survived column
#   (it's not in the features array)
# Still learning about ML, python, and kaggle

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from IPython.display import display

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# - consider training data to be _all_ data
all_data = pd.read_csv('titanic_mark_02_train.csv')
all_data.head()

y = all_data.Survived

features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
#features = ['Pclass', 'Sex', 'SibSp', 'Parch']
X = pd.get_dummies(all_data[features])
#X = train_data[features]
#print(X.describe()) # interesting

training_X, validation_X, training_y, validation_y = train_test_split(X, y, random_state=1)

model = DecisionTreeRegressor(random_state=1)

model.fit(training_X, training_y)

validation_predictions = model.predict(validation_X)
validation_MAE = mean_absolute_error(validation_predictions, validation_y)
print("Validation MAE: {:,.5f}".format(validation_MAE))

candidate_max_leaf_nodes = [5, 25, 50, 60, 63, 65, 66, 67, 68, 69, 70, 75, 85, 100, 150, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
min_mean_absolute_error = 1000000
for max_leaf_nodes in candidate_max_leaf_nodes:
    current_mean_absolute_error = get_mae(max_leaf_nodes = max_leaf_nodes, train_X=training_X, val_X=validation_X, train_y=training_y, val_y=validation_y)
    if current_mean_absolute_error < min_mean_absolute_error:
        min_mean_absolute_error = current_mean_absolute_error
        best_tree_size = max_leaf_nodes
print(f"best_tree_size: {best_tree_size}")

model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

model.fit(X, y)

predictions = model.predict(X)
#print(predictions[0])
predictions = [round(num) for num in predictions]
MAE = mean_absolute_error(predictions, y)
#print("MAE: {:,.5f}".format(MAE))
print("Accuracy: {:,.1f}%".format((1-MAE)*100))
#output = pd.DataFrame({'PassengerId': all_data.PassengerId, 'truth Survived': all_data.Survived, 'pred. Survived': predictions})
output = pd.DataFrame({'PassengerId': all_data.PassengerId, 'truth Survived': all_data.Survived, 'pred. Survived': predictions, 't/f:':all_data.Survived==predictions})
pd.options.display.max_rows = 4000 # show all rows in kaggle
print(output)
