import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter01/Dataset/dataset_44_spambase.csv'

df = pd.read_csv(file_url)
# print (df)

target = df.pop('class')
# print (target)

seed = 168
rf_model = RandomForestClassifier(random_state=seed)
rf_model.fit(df, target)
preds = rf_model.predict(df)
# print (preds)

acc_score = accuracy_score(target, preds)
print(acc_score)

