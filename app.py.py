import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import accuracy_score

data = {
    'Day': ['Sunny', 'Windy', 'Sunny', 'Windy', 'Windy', 'Sunny', 'Windy', 'Sunny', 'Sunny'], 
    'Temprature': ['Cool', 'Cool', 'Hot', 'Hot', 'Hot', 'Cool', 'Hot', 'Hot', 'Hot'], 
    'Class': ['Play', 'Not Play', 'Not Play', 'Play', 'Play', 'Play', 'Not Play', 'Play', 'Play']
}

df = pd.DataFrame(data)
x_raw = df[['Day','Temprature']]
y_raw = df['Class']

onehot_encoder = OneHotEncoder()
x_encoded = onehot_encoder.fit_transform(x_raw).toarray()
label_encoder = LabelEncoder()
label_encoded = label_encoder.fit_transform(y_raw)

x_train, x_test, y_train, y_test = train_test_split(x_encoded,label_encoded,test_size=0.3)

model = GaussianNB()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Accuraccy: ",accuracy_score(y_test,y_pred))

day = input("Enter Day Windy or Sunny: ")
temp = input("Enter Temprature Cool or Hot: ")
new_ins = pd.DataFrame([[day,temp]],columns=['Day','Temprature'])
new_ins_endoced = onehot_encoder.transform(new_ins).toarray()
predicted_class = model.predict(new_ins_endoced)
predicted_label = label_encoder.inverse_transform(predicted_class)[0]

print("Prediction for Windy and Hot Day is:",predicted_label)