## 1.Univariate CNN Models   
univariate : a single series of observations with a temporal ordering   
sequence -> input : 3 time steps, output : 1 time step for 1 step prediction   
### 1) Data Preparation   
```python
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D

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

print('(X, y)')

for i in range(len(X)):
  print(X[i], y[i])

n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features)) #[samples,timesteps]->[samples,timesteps,features]

for i in range(len(X)):
  print(X[i], y[i])
```
(X, y)   
[10 20 30] 40    
[20 30 40] 50    
[30 40 50] 60    
[40 50 60] 70    
[50 60 70] 80    
[60 70 80] 90    
[[10]    
 [20]    
 [30]] 40    
[[20]    
 [30]    
 [40]] 50    
[[30]    
 [40]    
 [50]] 60    
[[40]    
 [50]    
 [60]] 70    
[[50]    
 [60]    
 [70]] 80    
[[60]    
 [70]    
 [80]] 90    
### 2) CNN Model  
 ```python
model = Sequential()
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=1000, verbose=0)

# demonstrate prediction
x_input = array([70,80,90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```
[[100.82957]]
##   
## 2.Multivariate CNN Models
### 1) Multiple Input Series   
```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D

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

in_seq1 = array([10,20,30,40,50,60,70,80,90])
in_seq2 = array([15,25,35,45,55,65,75,85,95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
print(out_seq)

# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

dataset = np.hstack((in_seq1, in_seq2, out_seq))
print(dataset)

n_steps = 3
X, y = split_sequences(dataset, n_steps)
print(X.shape, y.shape)
for i in range(len(X)):
  print(X[i], y[i])

n_features = X.shape[2]
```
[ 25  45  65  85 105 125 145 165 185]   
[[ 10  15  25]   
 [ 20  25  45]   
 [ 30  35  65]   
 [ 40  45  85]   
 [ 50  55 105]   
 [ 60  65 125]   
 [ 70  75 145]   
 [ 80  85 165]   
 [ 90  95 185]]   
(7, 3, 2) (7,)   
[[10 15]   
 [20 25]   
 [30 35]] 65   
[[20 25]   
 [30 35]   
 [40 45]] 85   
[[30 35]   
 [40 45]   
 [50 55]] 105   
[[40 45]   
 [50 55]   
 [60 65]] 125   
[[50 55]   
 [60 65]   
 [70 75]] 145   
[[60 65]   
 [70 75]   
 [80 85]] 165   
[[70 75]   
 [80 85]   
 [90 95]] 185   
 
 ### 2) CNN Model   
 ```python
model = Sequential()
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=1000, verbose=0)

# demonstrate prediction
x_input = array([[80,85], [90,95], [100,105]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```
[[208.13545]]   
###    
#### Multiple Input - Multi-headed CNN Model
각 submodel의 output을 prediction이전에 combine해서 output sequence로 보냄
![Multi-headed](https://user-images.githubusercontent.com/63143652/83852211-6c086a80-a74e-11ea-94ed-5a399ff0e576.jpeg)

```python
from numpy import array, hstack
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate

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

in_seq1 = array([10,20,30,40,50,60,70,80,90])
in_seq2 = array([15,25,35,45,55,65,75,85,95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

in_seq1 = in_seq1.reshape((len(in_seq1), 1)) 
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

dataset = hstack((in_seq1, in_seq2, out_seq))

n_steps = 3
X, y = split_sequences(dataset, n_steps)
n_features = 1
```
### # separate input data
input을 2개로 분리해 각각의 submodel에 넣어주어야 한다.    
[7,3,2] => [7,3,1] x 2   
```python
X1 = X[:, :, 0].reshape(X.shape[0], X.shape[1], n_features)
X2 = X[:, :, 1].reshape(X.shape[0], X.shape[1], n_features)
# first input model
visible1 = Input(shape=(n_steps, n_features))
cnn1 = Conv1D(64, 2, activation='relu')(visible1)
cnn1 = MaxPooling1D(pool_size=2)(cnn1)
cnn1 = Flatten()(cnn1)
# second input model
visible2 = Input(shape=(n_steps, n_features))
cnn2 = Conv1D(64, 2, activation='relu')(visible2)
cnn2 = MaxPooling1D(2)(cnn2)
cnn2 = Flatten()(cnn2)
# merge input models
merge = concatenate([cnn1, cnn2])
dense = Dense(50, activation='relu')(merge)
output = Dense(1)(dense)
model = Model(inputs=[visible1, visible2], outputs=output)
model.compile(optimizer='adam', loss='mse')

model.fit([X1, X2], y, epochs=1000, verbose=0)

x_input = array([[80,85], [90,95], [100,105]])
x1 = x_input[:,0].reshape((1, n_steps, n_features))
x2 = x_input[:,1].reshape((1, n_steps, n_features))
yhat = model.predict([x1, x2], verbose=0)
print(yhat)
```
[[205.70218]]   
###
### 2) Multiple Parallel Series     
#### Multiple Parallel Series - Vector-output CNN Model    
#### Multiple Parallel Series - Multi-output CNN Model   
![IMG_68ED63A3FA6C-1](https://user-images.githubusercontent.com/63143652/83867317-30789b00-a764-11ea-9a9b-c6c28e9ea239.jpeg)
```python
from numpy import array, hstack
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D

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

in_seq1 = array([10,20,30,40,50,60,70,80,90])
in_seq2 = array([15,25,35,45,55,65,75,85,95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

in_seq1 = in_seq1.reshape((len(in_seq1), 1)) 
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

dataset = hstack((in_seq1, in_seq2, out_seq))

n_steps = 3
X, y = split_sequences(dataset, n_steps)
n_features = X.shape[2]

# separate output
y1 = y[:, 0].reshape((y.shape[0], 1))
y2 = y[:, 1].reshape((y.shape[0], 1))
y3 = y[:, 2].reshape((y.shape[0], 1))

# define model
visible = Input(shape=(n_steps, n_features))
cnn = Conv1D(64, 2, activation='relu')(visible)
cnn = MaxPooling1D(2)(cnn)
cnn = Flatten()(cnn)
cnn = Dense(50, activation='relu')(cnn)

output1 = Dense(1)(cnn)
output2 = Dense(1)(cnn)
output3 = Dense(1)(cnn)

model = Model(inputs=visible, outputs=[output1, output2, output3])
model.compile(optimizer='adam', loss='mse')

model.fit(X, [y1,y2,y3], epochs=2000, verbose=0)

x_input = array([[70,75,145], [80,85,165], [90,95,185]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```
[array([[99.81926]], dtype=float32), array([[109.228226]], dtype=float32), array([[210.63351]], dtype=float32)]   

##   
## 3.Multi-step CNN Models   
### 1) Data Preparation   
```python
from numpy import array

def split_sequence(sequence, n_steps_in, n_steps_out):
  X, y = list(), list()
  for i in range(len(sequence)):
    end_ix = i + n_steps_in
    out_end_ix = end_ix + n_steps_out
    if out_end_ix > len(sequence):
      break
    seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
    X.append(seq_x)
    y.append(seq_y)
  return array(X), array(y)

raw_seq = [10,20,30,40,50,60,70,80,90]
n_steps_in, n_steps_out = 3, 2
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)

for i in range(len(X)):
  print(X[i], y[i])
```
[10 20 30] [40 50]   
[20 30 40] [50 60]   
[30 40 50] [60 70]   
[40 50 60] [70 80]   
[50 60 70] [80 90]   
### 2) Vocetor Output Model   
```python
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

model = Sequential()
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=2000, verbose=0)

x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```
[[ 99.68114 116.26045]]    

## 4.Multivariate Multi-step CNN Models    
### 1) Multiple Input Multi-step Output   
```python
from numpy import array, hstack
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D

def split_sequences(sequences, n_steps_in, n_steps_out):
  X, y = list(), list()
  for i in range(len(sequences)):
    end_ix = i + n_steps_in
    out_end_ix = end_ix + n_steps_out-1
    if out_end_ix > len(sequences):
      break
    seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
    X.append(seq_x)
    y.append(seq_y)
  return array(X), array(y)

in_seq1 = array([10,20,30,40,50,60,70,80,90])
in_seq2 = array([15,25,35,45,55,65,75,85,95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

dataset = hstack((in_seq1, in_seq2, out_seq))
 
n_steps_in, n_steps_out = 3, 2
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
print(X.shape, y.shape)

for i in range(len(X)):
  print(X[i], y[i])
```
(6, 3, 2) (6, 2)   
[[10 15]   
 [20 25]   
 [30 35]] [65 85]   
[[20 25]   
 [30 35]   
 [40 45]] [ 85 105]   
[[30 35]   
 [40 45]   
 [50 55]] [105 125]   
[[40 45]   
 [50 55]   
 [60 65]] [125 145]   
[[50 55]   
 [60 65]   
 [70 75]] [145 165]   
[[60 65]   
 [70 75]   
 [80 85]] [165 185]   
```python
n_features = X.shape[2]

model = Sequential()
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=2000, verbose=0)

x_input = array([[70,75], [80,85], [90,95]])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```
[[185.45137 207.36356]]   

### 2) Multiple Parallel Input and Multi-step Output   
```python
from numpy import array, hstack
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D

def split_sequences(sequences, n_steps_in, n_steps_out):
  X, y = list(), list()
  for i in range(len(sequences)):
    end_ix = i + n_steps_in
    out_end_ix = end_ix + n_steps_out
    if out_end_ix > len(sequences):
      break
    seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
    X.append(seq_x)
    y.append(seq_y)
  return array(X), array(y)

in_seq1 = array([10,20,30,40,50,60,70,80,90])
in_seq2 = array([15,25,35,45,55,65,75,85,95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

dataset = hstack((in_seq1, in_seq2, out_seq))
 
n_steps_in, n_steps_out = 3, 2
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
print(X.shape, y.shape)

for i in range(len(X)):
  print(X[i], y[i])
```
(5, 3, 3) (5, 2, 3)   
[[10 15 25]   
 [20 25 45]   
 [30 35 65]] [[ 40  45  85]   
 [ 50  55 105]]   
[[20 25 45]   
 [30 35 65]   
 [40 45 85]] [[ 50  55 105]   
 [ 60  65 125]]   
[[ 30  35  65]   
 [ 40  45  85]   
 [ 50  55 105]] [[ 60  65 125]   
 [ 70  75 145]]   
[[ 40  45  85]   
 [ 50  55 105]   
 [ 60  65 125]] [[ 70  75 145]   
 [ 80  85 165]]   
[[ 50  55 105]   
 [ 60  65 125]   
 [ 70  75 145]] [[ 80  85 165]   
 [ 90  95 185]]   
```python
# flatten output
n_output = y.shape[1] * y.shape[2]
y = y.reshape((y.shape[0], n_output))
n_features = X.shape[2]

model = Sequential()
model.add(Conv1D(64, 2, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(n_output))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=2000, verbose=0)

x_input = array([[60,65,125], [70,75,145], [80,85,165]])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
```
[[ 90.74329  96.94914 188.3212  102.5601  107.99062 210.67068]]   
