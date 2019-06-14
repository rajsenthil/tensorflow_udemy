import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = np.random.randint(0, 100, (10, 2))
print(data)

scaler_model = MinMaxScaler()
scaler_model.fit(data)
scaler_model.transform(data)
print(scaler_model.transform(data))

import pandas as pd
mydata = data=np.random.randint(0,101,(50,4))
print(mydata)
df = pd.DataFrame(data=mydata, columns=['f1', 'f2', 'f3', 'label'])
print(df)

X=df[['f1', 'f2', 'f3']]
print(X)

y = df['label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train.shape)
print(X_test.shape)

