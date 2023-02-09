import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
# import categorical_embedder as ce
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import time
import os


# import shutil
# import  sys

def dir_is_empty(path):
    if len(os.listdir(path)) == 0:
        return True
    else:
        return False


# tensorboard --logdir="logs/"

# Folders preparations
# shutil.rmtree("logs\\fit")
# shutil.rmtree("models\\")
# os.mkdir("models")
# os.mkdir("logs\\fit")


# #Reading dataset
# dataset=pd.read_csv("res/SEER Breast Cancer Dataset .csv")

# Dataset analysis

# print(dataset.nunique())

# #X and Y data
# X=dataset.drop(["Status","Unnamed: 3"],axis=1)
# Y=dataset["Status"].replace(to_replace=["Alive","Dead"],value=[1,0])
#
# #Categorical embeddings
# embedding_info=ce.get_embedding_info(X)
# X_encoded,encoders=ce.get_label_encoded_data(X)
# embeddings = ce.get_embeddings(X_encoded, Y, categorical_embedding_info=embedding_info,
#                                is_classification=True, epochs=100, batch_size=256)
# X=ce.fit_transform(X,embeddings=embeddings,encoders=encoders,drop_categorical_vars=True)
#
# #Saving transformed dataset
# X.to_csv("res/X.csv",index=False)
# Y.to_csv("res/Y.csv", index=False)

# Visualisation
# fig,ax=plt.subplots(figsize=(20,20))
# sns.heatmap(dataset.corr(),annot=True)
# plt.show()

# Creating and fitting the model

# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs\\fit\\{}".format(time_stop))
# tensorboard = TensorBoard(log_dir="logs\\fit\\{}".format("Batch_norm"))

class BCDNeuralNetwork:
    def __init__(self, random_state=0, train_size=0.8):
        self.df=pd.read_csv("res/SEER Breast Cancer Dataset .csv")
        self.x = pd.read_csv("res/X.csv")
        self.y = pd.read_csv("res/Y.csv")
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, train_size=train_size,
                                                                                random_state=random_state)
        path = "models"
        if dir_is_empty(path):
            self.best_results = []
            self.best_model = None
        else:
            self.best_model = tf.keras.models.load_model(path + "/best_model")
            self.best_results = self.best_model.evaluate(self.x_test, self.y_test, batch_size=32)

    def shuffle(self, random_state, train_size=0.8):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, train_size=train_size,
                                                                                random_state=random_state)

    def train_model(self, times=1, epochs=30, validation_split=0.1, learning_rate=0.001):
        with tf.device("/CPU:0"):
            model = tf.keras.models.Sequential([
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(22, activation='relu'),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(11, activation='relu'),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # normal rate =0.001
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )

            for i in range(times):
                epochs = epochs
                model.fit(
                    x=self.x_train,
                    y=self.y_train,
                    validation_split=validation_split,
                    epochs=epochs,
                    batch_size=32,
                )

                results = model.evaluate(self.x_test, self.y_test, batch_size=32)
                if not self.best_results:
                    self.best_results = results
                    self.best_model = model
                else:
                    if results[1] > self.best_results[1]:
                        print(f"NEW NEURAL {results[1]}>{self.best_results[1]} OF OLD NEURAL - CHANGE!")
                        self.best_results = results
                        self.best_model = model
                    else:
                        print(f"NEW NEURAL {results[1]}<{self.best_results[1]} OF OLD NEURAL - NO CHANGE")
            path = "models/best_model"
            if os.path.isfile(path):
                os.remove(path)
            self.best_model.save("models/best_model")

    def match_unique_values(self,embedded_columns,matching_column,value):
        cut_embedded_columns=self.x[embedded_columns].drop_duplicates()
        #print(cut_embedded_columns.head())
        cut_matching_columns =self.df[matching_column].drop_duplicates().to_frame()
        #print(cut_matching_columns.head())
        for index,row in cut_matching_columns.iterrows():
            if(row.values==value):
                #print(f"FOUND IT {index}")
                value_index=index
                #print(f"THE MATCH IS {cut_embedded_columns.loc[value_index].values}")
                return cut_embedded_columns.loc[value_index].values
            #print(row.values)

    def get_list(self,column):
        l=self.df[column].drop_duplicates().to_frame().values.tolist()
        #print(list(map(lambda x:x[0],l)))
        return list(map(lambda x:x[0],l))



