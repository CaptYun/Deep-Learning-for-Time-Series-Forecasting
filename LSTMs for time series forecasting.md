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
```python
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional

model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=200, verbose=0)

x_input = array([70,80,90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```
[[101.653656]]   

### 5. CNN-LSTM   
CNN은 1D sequnce data (univariate time series data)에서 features를 뽑아 학습하기 유용하다.   
input data : [samples, subsequences, timesteps, features]   
###   
1.univariate time series data를 input(4steps)/output(1)으로 나눔    
2.각각 2 time steps을 가진 2개의 sub-samples로 나눔    
3.CNN으로 각각의 subsequence를 interpret하여 LSTM에 input으로 전달   
*TimeDistributed : 각 input(subsequence)당 한번 전체 모델 적용   
```python
n_steps = 4  #number of time steps
X, y = split_sequence(raw_seq, n_steps)

#reshape [samples, timesteps]->[samples, subsequences, timesteps, features]
n_features = 1
n_seq = 2    #number of time steps / n_steps
n_steps = 2  
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))

from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, TimeDistributed
from keras.layers.convolutional import Conv1D, MaxPooling1D

# input CNN model
model = Sequential()
model.add(TimeDistributed(Conv1D(64, 1, activation='relu'), 
                          input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
# output model
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=500, verbose=0)

x_input = array([60,70,80,90])
x_input = x_input.reshape((1, n_seq, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```
[[100.72959]]   
### 6. ConvLSTM   
2D spatial-temporal data를 읽을 때 유용   
input data : [samples, timesteps, rows, columns, features]    
kernel size : 2D (rows, columns), 1D series일 경우 rows=1     
```python
n_steps = 4  #number of time steps
X, y = split_sequence(raw_seq, n_steps)

#reshape [samples, timesteps]->[samples, subsequences, timesteps, features]
n_features = 1
n_seq = 2    #number of time steps / n_steps
n_steps = 2  
X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))

from keras.models import Sequential
from keras.layers import Dense, Flatten, ConvLSTM2D


model = Sequential()
model.add(ConvLSTM2D(64, kernel_size=(1,2), activation='relu', 
                          input_shape=(n_seq, 1, n_steps, n_features)))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=500, verbose=0)

x_input = array([60,70,80,90])
x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```
[[104.25568]]   
##   
## Multivariate LSTM Models   
### 1. Multiple Input Series   
2개 이상의 parallel input time series와 input에 종속된 1개의 output time series 문제   
input time series는 parallel하다.(같은 time steps에서 측정된 값이기 때문)   
input shape = [samples, time steps, parallel time series(variables)]    
```python
from numpy import array, hstack

def split_sequences(sequences, n_steps):
  X, y = list(), list()
  for i in range(len(sequences)):
    end_ix = i + n_steps
    if end_ix > len(sequences):
      break
    seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
    X.append(seq_x)
    y.append(seq_y)
  return array(X), array(y)

# define input sequence   
in_seq1 = array([10,20,30,40,50,60,70,80,90])
in_seq2 = array([15,25,35,45,55,65,75,85,95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure  
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
dataset = hstack((in_seq1, in_seq2, out_seq))
print('in_seq1\n',in_seq1)
print('dataset\n',dataset)
n_steps = 3
X, y = split_sequences(dataset, n_steps)
print('X.shape=',X.shape,'\ny.shape=',y.shape)
for i in range(len(X)):
  print(X[i], y[i])
```
<img width="168" alt="스크린샷 2020-06-06 10 26 03" src="https://user-images.githubusercontent.com/63143652/83933059-56438580-a7e0-11ea-94f9-81b8f82733ac.png">
<img width="159" alt="스크린샷 2020-06-06 10 26 13" src="https://user-images.githubusercontent.com/63143652/83933056-53e12b80-a7e0-11ea-969a-31038b793f2b.png">

```python
n_features = X.shape[2]  # num_parallel time series   

from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, verbose=0)

x_input = array([[80,85],[90,95],[100,105]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```

### 2. Multiple Parallel Series   
```python
from numpy import array, hstack

def split_sequences(sequences, n_steps):
  X, y = list(), list()
  for i in range(len(sequences)):
    end_ix = i + n_steps
    if end_ix >= len(sequences):
      break
    seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
    X.append(seq_x)
    y.append(seq_y)
  return array(X), array(y)

# define input sequence   
in_seq1 = array([10,20,30,40,50,60,70,80,90])
in_seq2 = array([15,25,35,45,55,65,75,85,95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure  
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
dataset = hstack((in_seq1, in_seq2, out_seq))

n_steps = 3
X, y = split_sequences(dataset, n_steps)
print('X.shape=',X.shape,'\ny.shape=',y.shape)
for i in range(len(X)):
  print(X[i], y[i])
```
<img width="254" alt="스크린샷 2020-06-06 10 33 09" src="https://user-images.githubusercontent.com/63143652/83933168-2f398380-a7e1-11ea-96a4-d5561c049968.png">
RNN에서 return_sequences : https://blog.naver.com/acelhj/221278800093   
```python
n_features = X.shape[2]  # num_parallel time series   

from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, #LSTM층을 쌓으려면 True
               input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=400, verbose=0)

x_input = array([[70,75,145],[80,85,165],[90,95,185]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```
[[100.80308 105.90578 206.38626]]   
## Multi-step LSTM Models   

## Multivariate Multi-step LSTM Models   
