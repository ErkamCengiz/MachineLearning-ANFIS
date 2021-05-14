import myanfis
import pandas as pd
import numpy
import sys

sys.maxsize
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# veriyi ekledik

df = pd.read_csv("winequality-red (1).csv")
# veriye bir bakış attık
df.head()
# verideki bütün değerlerin numeric olduğunu gördük
df.describe()

# oluşturacağımız model için kullanacağımız feature'lar "volatite acidity" ve "density"
# oluşturacağımız model için target değerlerimiz ise quality'den gelecek
# model için verilerimi 0-1 arasına yerleştirmemiz gerekiyor bunun için MinMaxScaler kullanacağız
minmaxScaler = MinMaxScaler()

# verimizin bir adet kopyasını aldık ki yaptığımız değişiklikler df'i etkilemesin
df2 = df.copy()

# aşağıdaki kısımdaysa verilerimizi 0-1 arasına yerleştiriyoruz
df2['volatile acidity'] = minmaxScaler.fit_transform(df2[['volatile acidity']])
df2['density'] = minmaxScaler.fit_transform(df2[['density']])
df2['quality'] = minmaxScaler.fit_transform(df2[['quality']])

# training ve validation için 2 ye ayırıyoruz
# training
X = df2.iloc[:-199, [1, 7]]
Y = df2.iloc[:-199, -1]

# validation, burayı kullanmıyoruz
x = df2.iloc[-199:, [1, 7]]
y = df2.iloc[-199:, -1]

# model için gerekli parametre tanımlarını burada "fis_parameters" class'ı ile yapıyoruz
param = myanfis.fis_parameters(
    n_input=2,  # no. of Regressors
    n_memb=2,  # no. of fuzzy memberships
    batch_size=5,  # 16 / 32 / 64 / ...
    memb_func='gaussian',  # 'gaussian' / 'gbellmf' / 'sigmoid'
    optimizer='sgd',  # sgd / adam / ...
    loss=tf.keras.losses.MeanAbsoluteError(),  # mse / mae / huber_loss / mean_absolute_percentage_error / ...
    n_epochs=15  # 10 / 25 / 50 / 100 / ...
)

# Bu görevde bir supervised yöntem olduğu için KFold = 2 olarak belirledik
kfold = KFold(n_splits=2)
histories = []

# Fold'larımız için gerekli indexleri kfold.split ile alarak X_train, X_test, Y_train, Y_test olarak aldık
# ve Fold Fold eğitimini gerçekleştirdik
for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Tensorflow.Keras modellerinden üretilmiş ANFIS'i gerekli parametreler ile çağırdık
    fis = myanfis.ANFIS(n_input=param.n_input,
                        n_memb=param.n_memb,
                        batch_size=param.batch_size,
                        memb_func=param.memb_func,
                        name='firstAnfis'  # buradaki ismi değiştirin
                        )

    # modelimizi aşağıdaki parametreler ile derledik
    fis.model.compile(optimizer=param.optimizer,
                      loss=param.loss,
                      metrics=['mae']  # ['mae', 'mse']
                      )
    # burada modelimizin eğitimini başlattık
    # daha sonrasında modelin eğitim sonuçlarını history'e atama yaptık
    history = fis.fit(X_train, Y_train,
                      epochs=param.n_epochs,
                      batch_size=param.batch_size,
                      validation_data=(X_test, Y_test),
                      )
    # model fold fold eğitim yapacağı için eğitim sonuçlarını histories adlı listede tuttuk
    histories.append(history)
# model eğitiminde elde edilen membership functions grafikleri şekildeki gibidir
fis.plotmfs()

# ilk iterasyonda elde edilen eğitim grafiği aşağıdaki şekilde olduğu gibidir
# İlk fold için konuşacak olursak eğer model iyi bir eğitim yapmadığını söyleyebilirz çünkü val_mae, loss'un üstünde kalmıi
pd.DataFrame(histories[0].history).plot()

# ikinci iterasyonda elde edilen eğitim grafiği aşağıdaki şekilde olduğu gibidir
# İkinci fold için konuşacak olrusak burada model ilk folda göre oldukça daha başarılı olduğunu söylebiliriz
# çükü val_mae loss çizgisinin altında kalmış durumda

pd.DataFrame(histories[1].history).plot()