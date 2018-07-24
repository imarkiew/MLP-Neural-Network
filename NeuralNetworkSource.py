import math
import pandas as pd
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

#Rozmiar pakietu do uczenia
batch_size = 512
#Liczba epok
training_epochs = 200
#Dwie warstwy ukryte z dropout i funkcja aktywacji elu, na wyjsciu funkcja sigmoidalna, unipolarna
#Liczba neuronow w poszczegolnych warstwach ukrytych
hiddens = [256, 256]
#dropout dla warstw ukrytych, metoda regularyzacji polegajaca na braku brania pod uwage niektorych neuronow w sposob
#losowy, majaca ogrraniczyc prawdopodobienstwo nadmiernego dopasowania
dropout_prob = 0.5
#wspolczynnik szybkosci uczenia
learning_rate = 0.001
#procent liczby przykladow przeznaczonych do testowania
percent_of_test_examples = 0.3

df = pd.read_csv("./Data/data.csv", index_col="Id")
#zastepowanie wyrazen "NA" srednimi arytmetycznymi po danej kolumnie
df = df.fillna(value=df.mean())
#cel klasyfikacji binarnej
y_column = "SeriousDlqin2yrs"
y = df[y_column].values
df = df.drop(y_column, axis=1)
#normalizacja danych wejsciowych
df_norm = (df - df.mean()) / (df.max() - df.min())
X = df_norm.values
#Podzial na zbiory uczace i testowe (klasyfikacje osobno)
Xx, Xt, yy, yt = train_test_split(X, y, test_size=percent_of_test_examples, stratify=y)
#nadprobkowanie danych pozytywnych przy pomocy algorytmu k-srednich w celu zrownowazenia kategorii
#Przynosi czesto lepsze efekty niz podprobkowanie
smt = SMOTE()
Xxr, yyr = smt.fit_sample(Xx, yy)
Xtr, ytr = smt.fit_sample(Xt, yt)

#Rozmiar wyjscia, klasyfikacja binarna => rozmiar = 1
out_dim = 1
#Rozmiar wejscia
in_dim = X.shape[1]

x = tf.placeholder("float", [None, in_dim])
y = tf.placeholder("float", [None, out_dim])
keep_prob = tf.placeholder("float")
dim = [in_dim, *hiddens, out_dim]
weights = [tf.Variable(tf.random_normal([dim[i-1], dim[i]])) for i in range(1, len(dim))]
biases = [tf.Variable(tf.constant(0.1, shape=[dim[i]])) for i in range(1, len(dim))]

#Funkcja do tworzenia modelu sieci
def create_model(x, weights, biases, keep_prob):
    layer = x
    for i in range(len(dim) - 1):
        layer = tf.nn.xw_plus_b(layer, weights[i], biases[i])
        if i < len(hiddens):
            layer = tf.nn.elu(layer)
            layer = tf.nn.dropout(layer, keep_prob=keep_prob)
        else:
            layer = tf.nn.sigmoid(layer)
    return layer

#Wyjscie modelu sieci
model_out = create_model(x, weights, biases, keep_prob)
#Funkcja kosztu na podstawie zmodyfikowanej normy l2
loss = tf.nn.l2_loss(y - model_out)
#Sposob optymalizacji Adam(metoda adaptacyjna) - dobre wyniki, choc czasem moze utknac w minimach lokalnych
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#Generator pakietow do uczenia
def generate_batches(X, y):
    X, y = shuffle(X, y)
    batches_num = math.ceil(y.shape[0] / batch_size)
    for i in range(0, batches_num):
        yield X[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]

#Wektor wartosci funkcji kosztu dla uczenia
train_losses = []
#Wektor wartosci funkcji kosztu dla walidacji
val_losses = []

#Otwarcie sesji
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    val_loss = 0

    for epoch in range(training_epochs):
        tl = []
        #Uczenie i blad dla uczenia
        for i, (batch_x, batch_y) in enumerate(generate_batches(Xxr, yyr)):
            tl.append(
                sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y.reshape(-1, 1), keep_prob: dropout_prob})[
                    1])
        train_loss = np.mean(tl)

        #Blad dla zbioru testowego
        vl = []
        for i, (batch_x, batch_y) in enumerate(generate_batches(Xtr, ytr)):
            vl.append(sess.run(loss, feed_dict={x: batch_x, y: batch_y.reshape(-1, 1), keep_prob: 1.0}))

        val_loss = np.mean(vl)

        print("Epoch: {} loss {} validation loss {}".format(epoch + 1, train_loss, val_loss))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    #Wyjscie sieci dla zbioru testowego
    results = model_out.eval(feed_dict={x: Xtr, keep_prob: 1.0})

    count_positive = 0
    #Sprawdzanie celnosci predykcji dla zbioru testowego
    for j in range(0, len(ytr)):
        if (ytr[j] == 1 and results[j] >= 0.5) or (ytr[j] == 0 and results[j] < 0.5):
            count_positive += 1
    print("Accuracy " + str((count_positive/len(ytr))*100))

    #Rysowanie wykresu bledu na zbiorze uczacym i testowym
    line = range(len(train_losses))
    fig = plt.figure()
    ax = plt.subplot()
    ax.plot(line, train_losses, "b+", label="Blad na zbiorze uczacym")
    ax.plot(line, val_losses, "r-", label="Blad na zbiorze testowym")
    plt.xlabel('i', fontsize=15)
    plt.ylabel('blad(i)', fontsize=15)
    ax.legend()
    fig.savefig('256_256_0.001_200.png', bbox_inches='tight')
    plt.show()
    input()

