# Импортирование необходимых библиотек
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from joblib import dump
from feature_engine.creation import MathFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from tabgan.sampler import ForestDiffusionGenerator

basedir = os.path.dirname(__file__)

# Класс Data для подготовки данных к обучению
class Data:
    def __init__(self, data_filepath, need_extend=True):
        self.X_train, self.X_test, self.y_train, self.y_test, self.features_encoder = self.__prepareData(data_filepath, need_extend)

    def __prepareData(self, data_filepath, need_extend):
        df_features = self.__readData(data_filepath)
        df_features = self.__addNewFeatures(df_features, variables=["Балл по математике", "Балл по русскому языку", "Балл по выбранному предмету"], func = "mean", new_variables_names=["Средний балл по экзаменам"])
        df_features = self.__addNewFeatures(df_features, variables=["Балл по математике", "Балл по русскому языку", "Балл по выбранному предмету", "Балл за индивидуальные достижения"], func = "sum", new_variables_names=["Сумма баллов"])
        df_features = self.__addNewFeatures(df_features, variables=["Балл по математике", "Балл по русскому языку", "Балл по выбранному предмету", "Балл за индивидуальные достижения"], func = lambda scores: (sum(scores) / 347) * 100, new_variables_names=["Процент от маскимального балла"])
        df_target = self.__clusteringFeatures(df_features, idx=[3,0,1,2])
        if need_extend:
                df_features = self.__extendingFeatures(df_features, df_target)
                df_features = self.__addNewFeatures(df_features, variables=["Балл по математике", "Балл по русскому языку", "Балл по выбранному предмету"], func = "mean", new_variables_names=["Средний балл по экзаменам"])
                df_features = self.__addNewFeatures(df_features, variables=["Балл по математике", "Балл по русскому языку", "Балл по выбранному предмету", "Балл за индивидуальные достижения"], func = "sum", new_variables_names=["Сумма баллов"])
                df_features = self.__addNewFeatures(df_features, variables=["Балл по математике", "Балл по русскому языку", "Балл по выбранному предмету", "Балл за индивидуальные достижения"], func = lambda scores: (sum(scores) / 347) * 100, new_variables_names=["Процент от маскимального балла"])
                df_target = self.__clusteringFeatures(df_features, idx=[3,1,2,0])
        X = np.array(df_features)
        y = np.array(df_target)
        ohe_y = OneHotEncoder()
        y_cat = ohe_y.fit_transform(np.expand_dims(y.flatten(),axis=-1)).toarray()
        X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
        features_encoder = StandardScaler()
        features_encoder.fit(X_train)
        X_train = features_encoder.transform(X_train)
        X_test = features_encoder.transform(X_test)
        return X_train, X_test, y_train, y_test, features_encoder

    def __readData(self, data_filepath):
        return pd.read_csv(data_filepath, delimiter=";")

    def __addNewFeatures(self, df_features, variables, func, new_variables_names):
        mf = MathFeatures(variables=variables, func=func, new_variables_names=new_variables_names)
        mf.fit(df_features)
        return mf.transform(df_features)

    def __clusteringFeatures(self, df_features, idx):
        ac = AgglomerativeClustering(n_clusters=4).fit(df_features)
        df_target = pd.DataFrame(ac.labels_, columns=["Успеваемость"])
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(4)
        df_corrected_target = pd.DataFrame(lut[df_target], columns=["Успеваемость"])
        return df_corrected_target
    
    def __extendingFeatures(self, df_features, df_target):
        df_extended_features, _ = ForestDiffusionGenerator(gen_x_times=100, cat_cols=["Пол", "Приоритет", "Наличие серебрянной медали", "Наличие золотой медали"], adversarial_model_params={'learning_rate': 0.02, 'random_state': 42}, gen_params={'batch_size': 30, 'epochs': 100}).generate_data_pipe(df_features.drop(columns=["Средний балл по экзаменам", "Сумма баллов", "Процент от маскимального балла"]), df_target, df_features.drop(columns=["Средний балл по экзаменам", "Сумма баллов", "Процент от маскимального балла"]))
        return df_extended_features

# Класс KerasModel для создания, обучения, валидации и использования нейронной сети
class KerasModel(BaseEstimator, ClassifierMixin):
    def __init__(self, X_train, y_train, X_test, y_test, layers, optimizer, loss, metrics):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.model = tf.keras.models.Sequential(self.layers)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
    
    def fit(self, X_train, y_train, n_epoch):
        self.history = self.model.fit(x=X_train, y=y_train, epochs=n_epoch, validation_data=(self.X_test, self.y_test), verbose=0)
        return self.model

    def summary(self):
        return self.model.summary()

    def evaluate(self, X_test, y_test):
        return self.model(X_test, y_test)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Error (Loss)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_accuracy(self):
        plt.plot(self.history.history['categorical_accuracy'], label='accuracy')
        plt.plot(self.history.history['val_categorical_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

data = Data(os.path.join(basedir, 'data.csv'), need_extend=True)
model = KerasModel(X_train = data.X_train, 
                   y_train = data.y_train, 
                   X_test = data.X_test, 
                   y_test = data.y_test, 
                   layers = [tf.keras.layers.Input((data.X_train.shape[1], )),
                             tf.keras.layers.Dense(10,activation='relu'),
                             tf.keras.layers.Dense(10,activation='relu'),
                             tf.keras.layers.Dense(4, activation="softmax"),],
                   optimizer = tf.keras.optimizers.SGD(),
                   loss = tf.keras.losses.CategoricalCrossentropy(), 
                   metrics = [tf.keras.metrics.CategoricalAccuracy(), 
                              tf.keras.metrics.AUC(curve='PR', name='pr_auc_accuracy'), 
                              tf.keras.metrics.AUC(curve='ROC', name='roc_auc_accuracy')])

model.fit(data.X_train, data.y_train, n_epoch=25)
yp=np.argmax(model.predict(data.X_test),axis=-1)
yt=np.argmax(data.y_test,axis=-1)
print(confusion_matrix(yt,yp))
print(classification_report(yt,yp))

model.model.export('model')
dump(data.features_encoder, 'std_scaler.bin', compress=True)