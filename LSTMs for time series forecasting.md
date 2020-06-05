## Univariate LSTM Models   
###  
### 1. Data Preparation   
```python
from numpy import array

def split_sequence(sequence, n_steps):
  X, y = list(), list()
  for i in range(len(sequence)):
    end_ix = i + n_steps
    if end_ix >= len(sequence):
      break
    seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
    X.append(seq_x)
    y.append(seq_y)
  return array(X), array(y)

raw_seq = [10,20,30,40,50,60,70,80,90]
n_steps = 3
X, y = split_sequence(raw_seq, n_steps)
for i in range(len(X)):
  print(X[i], y[i])
```
[10 20 30] 40   
[20 30 40] 50   
[30 40 50] 60   
[40 50 60] 70   
[50 60 70] 80   
[60 70 80] 90   
 
```python   
# define input sequence
raw_seq = [10,20,30,40,50,60,70,80,90]
n_steps = 3
n_features = 1
X, y = split_sequence(raw_seq, n_steps)
X = X.reshape((X.shape[0], X.shape[1], n_features))
```
### 2. Vanilla LSTM  
[samples, timesteps] -> training data [samples, timesteps, features]   
```python
from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=200, verbose=0)

x_input = array([70,80,90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```
[[101.74156]]   
### 3. Stacked LSTM  
return_sequence=True : 3D output을 다음 층에 input으로 전달   
```python
from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=200, verbose=0)

x_input = array([70,80,90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```
[[102.119804]]   
### 4. Bidirectional LSTM   
양방향에서 학습한 interpretations을 concatenate함   
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))   
### 5. CNN-LSTM   
CNN은 1D sequnce data (univariate time series data)에서 features를 뽑아 학습하기 유용하다.   
input data : [samples, subsequences, timesteps, features]   
## Multivariate LSTM Models   

## Multi-step LSTM Models   

## Multivariate Multi-step LSTM Models   
