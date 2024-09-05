import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from numpy.random import seed
seed(8)
import tensorflow as tf
import numpy as np
from sklearn import datasets, model_selection
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
import pickle
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def one_hot_encoding(df):
    '''Funcion para co'''
    dummies = pd.get_dummies(df['primary_genre'], dtype=float)
    df = pd.concat([df, dummies], axis=1)
    df.drop('primary_genre', axis=1, inplace=True)
    return df

def scaling(X):
    """Estandarización (escalado) de columnas a partir del índice 8 hacia adelante"""
    means = X.mean()
    stds = X.std()
    new_X = (X - means) / stds
    return new_X

def split(df):

    # Se mueve el df de forma aleatoria para eliminar ordenamientos y brindar aleatoriedad
    np.random.seed(42)
    df = df.sample(frac=1)

    # Se definen los atributos y el label
    X = df.drop('rating', axis=1)
    y = df['rating']


    # Se separa en train y test, 80% y 20% respectivamente
    split_index = int(0.8 * len(df))
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test


def graph(df):
    """Funcion para graficar los atributos contra rating"""
    num_columns = 9
    fig, axes = plt.subplots(5, 2, figsize=(10, 10))
    axes = axes.flatten()
    for i, column in enumerate(df.select_dtypes(include=['number']).columns):
        if column != 'rating':
            axes[i].scatter(df[column], df['rating'])
            axes[i].set_title(f'{column} vs rating')
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('rating')


    plt.tight_layout()
    plt.show()


def clean_data(df):
    """Funcion para evaluar, eliminar y transformar datos iniciales del df"""

    # Eliminar columnas
    df.drop(['Unnamed: 0', "game", "link", "release", "store_asset_mod_time", "detected_technologies", "all_time_peak_date", "store_genres", "publisher", "developer"], axis=1, inplace=True)

    df['players_right_now'] = df['players_right_now'].str.replace(',', '').astype(float)
    df['24_hour_peak'] = df['24_hour_peak'].str.replace(',', '').astype(float)

    #df = df.fillna(df["review_percentage"].mean())
    df.drop(df[df['review_percentage'].isna()].index, inplace=True)

    df.drop(df[df['players_right_now'].isna()].index, inplace=True)
    df.drop(df[df['24_hour_peak'].isna()].index, inplace=True)
    #print(df.isna().sum())

    #graph(df)

    df.drop(df[(df['peak_players'] <= 0) | (df['peak_players'] > 1000000)].index, inplace=True)
    df.drop(df[df['negative_reviews'] > 200000].index, inplace=True)
    df.drop(df[df['players_right_now'] > 100000].index, inplace=True)
    df.drop(df[df['all_time_peak'] > 500000].index, inplace=True)
    df.drop(df[df['positive_reviews'] > 1000000].index, inplace=True)
    df.drop(df[df['total_reviews'] > 1000000].index, inplace=True)
    df.drop(df[df['24_hour_peak'] > 200000].index, inplace=True)
    df = one_hot_encoding(df)
    return df

#Modelo

def get_model(input_shape):
    """
    Función que construye un modelo secuencial para un problema de regresión.
    """
    model = Sequential([
        Dense(64, input_shape=input_shape,  # Número_de_características
              activation="relu",
              kernel_regularizer=regularizers.l2(.01),
              kernel_initializer=tf.keras.initializers.HeUniform(),
              bias_initializer="ones"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        # Capa de salida para regresión: 1 neurona, activación lineal
        Dense(1, activation="linear")
    ])
    return model

def compile_model(model):
    """
    Función que toma el modelo devuelto por su función get_model y lo compila con un optimizador,
    función de pérdida y métrica.
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss="mean_squared_error",  # MSE
                  metrics=["mae"])  # Métrica de regresión

def train_model(model, train_data, train_targets, epochs):
    """
    Función que entrena el modelo para el número dado de épocas en el
    train_data y train_targets.
    Su función debería devolver el historial de entrenamiento.
    """
    return model.fit(train_data, train_targets, epochs = epochs, validation_split = 0.15, batch_size = 40)


def main():


    # Se guardan los datos en un dataframe
    df = pd.read_csv('game_data_all.csv')

    df = clean_data(df)

    X_train, X_test, y_train, y_test = split(df)

    X_train = scaling(X_train)
    X_train.to_numpy()
    y_train.to_numpy()


    X_test = scaling(X_test)
    X_test.to_numpy()
    X_test = X_test.fillna(0)
    y_test.to_numpy()

    model = get_model(X_train.shape)
    compile_model(model)

    input_shape = (X_train.shape[1],)  # Número de características
    model = get_model(input_shape)
    compile_model(model)
    history = train_model(model, X_train, y_train, epochs=100)

    # Guardar el historial en un archivo pickle
    with open('historial_entrenamiento_dos.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    # Graficar el error (pérdida) vs. épocas
    plt.plot(history.history['loss'], label='Pérdida (Training Loss)')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Pérdida de validación (Validation Loss)')
    plt.title('Pérdida vs Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()


    # Gráfica de MAE (Mean Absolute Error)
    if 'mae' in history.history:
        plt.plot(history.history['mae'], label='MAE en Entrenamiento (Training MAE)')
        if 'val_mae' in history.history:
            plt.plot(history.history['val_mae'], label='MAE de Validación (Validation MAE)')
        plt.title('Error Absoluto Medio (MAE) vs Épocas')
        plt.xlabel('Épocas')
        plt.ylabel('MAE')
        plt.legend()
        plt.show()

    # Hacer las predicciones para test
    y_pred = model.predict(X_test)

    # Calcular el R^2
    r2 = r2_score(y_test, y_pred)
    print(f"R2: {r2:.4f}")

    # Obtener las predicciones para el conjunto de validación
    y_pred_val = model.predict(X_train[int(0.85 * len(X_train)):])

    # Calcular el R^2 para el conjunto de validación
    r2_val = r2_score(y_train[int(0.85 * len(y_train)):], y_pred_val)

    # Graficar el R^2 para el conjunto de validación y test 
    plt.figure(figsize=(8, 6))
    plt.bar(['Validation', 'Test'], [r2_val, r2], color=['skyblue', 'orange']) 
    plt.ylabel('R-squared')
    plt.title('R-squared para conjuntos de validación y test')
    plt.show()

    print(f"R2 para el conjunto de validación: {r2_val:.4f}")
    print(f"R2 para el conjunto de prueba: {r2:.4f}")

    #ARBOL
    tree_reg = DecisionTreeRegressor(max_depth=10)
    tree_reg.fit(X_train, y_train)
    y_pred_tree = tree_reg.predict(X_test)
    print("R2 arbol de decision", r2_score(y_test, y_pred_tree))

    rnd_reg = RandomForestRegressor(n_estimators=400, max_leaf_nodes=10, n_jobs=-1, random_state=42)
    rnd_reg.fit(X_train, y_train)

    X_test = X_test.fillna(0)


    y_pred_rf= rnd_reg.predict(X_test)
    print("R2 random forest", r2_score(y_test, y_pred_rf))


main()
#Agregar matriz de confusion