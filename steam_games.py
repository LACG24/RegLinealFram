import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.random import seed
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
import pickle
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Configuración de semillas para reproducibilidad
seed(8)
np.random.seed(8)
tf.random.set_seed(8)

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
    """Función para separar el dataset en features y label y en test y train"""

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
        Dense(64, input_shape=input_shape, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
        Dense(256, activation="relu"),
        Dropout(0.20),
        Dense(128, activation="relu"),
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
    """
    return model.fit(train_data, train_targets, epochs = epochs, validation_split = 0.15, batch_size = 40)

def cross_val_plot(X_train, y_train, model_type='tree', param_grid=None, n_iter=10, random_state=42):
    """Funcion para validacion cruzada implementada para arboles ID3 y random forest"""
    if model_type == 'tree':
        model = DecisionTreeRegressor(random_state=random_state)
        search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1) if param_grid else RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=5, scoring='r2', n_jobs=-1, random_state=random_state)
    elif model_type == 'rf':
        model = RandomForestRegressor(random_state=random_state)
        search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1) if param_grid else RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=5, scoring='r2', n_jobs=-1, random_state=random_state)
    else:
        raise ValueError("model_type must be 'tree' or 'rf'")

    try:
        search.fit(X_train, y_train)
    except ValueError as e:
        print(f"Error during model fitting: {e}")
        return

    results = search.cv_results_
    mean_scores = results['mean_test_score']
    params = results['params']

    if model_type == 'tree':
        mean_scores = np.array(mean_scores).reshape(len(param_grid['max_depth']), len(param_grid['min_samples_split']))
        plt.figure(figsize=(10, 6))
        for i, min_samples in enumerate(param_grid['min_samples_split']):
            plt.plot(param_grid['max_depth'], mean_scores[:, i], marker='o', label=f'min_samples_split={min_samples}')
        plt.xlabel('max_depth')
        plt.ylabel('Mean Test R2 Score')
        plt.title('Validation Curve for Decision Tree')
        plt.legend()
    elif model_type == 'rf':
        mean_scores = np.array(mean_scores).reshape(len(param_grid['n_estimators']),
                                                     len(param_grid['max_depth']),
                                                     len(param_grid['max_features']),
                                                     len(param_grid['min_samples_split']))
        plt.figure(figsize=(14, 6))
        for i, n_estimators in enumerate(param_grid['n_estimators']):
            plt.plot(param_grid['max_depth'], mean_scores[i, :, 0], marker='o', label=f'n_estimators={n_estimators}')
        plt.xlabel('max_depth')
        plt.ylabel('Mean Test R² Score')
        plt.title('Validation Curve for Random Forest')
        plt.legend()

    plt.show()

def main():
    df = pd.read_csv('game_data_all.csv')
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split(df)
    X_train = scaling(X_train)
    X_test = scaling(X_test)
    X_test = X_test.fillna(0)
    
    input_shape = (X_train.shape[1],)
    model = get_model(input_shape)
    compile_model(model)
    
    max_epochs = 150
    history = train_model(model, X_train, y_train, epochs=max_epochs)

    plt.plot(history.history['loss'], label='Pérdida (Training Loss)')
    plt.plot(history.history['val_loss'], label='Pérdida de validación (Validation Loss)')
    plt.title('Pérdida vs Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()

    if 'mae' in history.history:
        plt.plot(history.history['mae'], label='MAE en Entrenamiento (Training MAE)')
        plt.plot(history.history['val_mae'], label='MAE de Validación (Validation MAE)')
        plt.title('Error Absoluto Medio (MAE) vs Épocas')
        plt.xlabel('Épocas')
        plt.ylabel('MAE')
        plt.legend()
        plt.show()

    # Red neuronal
    y_pred_test = model.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    val_data = X_train[int(0.85 * len(X_train)):]
    val_targets = y_train[int(0.85 * len(y_train)):]
    y_pred_val = model.predict(val_data)
    r2_val = r2_score(val_targets, y_pred_val)

    y_pred_train = model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)

    plt.figure(figsize=(10, 6))
    plt.bar(['Training', 'Validation', 'Test'], [r2_train, r2_val, r2_test], color=['skyblue', 'green', 'orange'])
    plt.ylabel('R-squared')
    plt.title('R-squared para conjuntos de entrenamiento, validación y test (Red neuronal)')
    plt.show()

    print(f"R2 para el conjunto de entrenamiento (Red neuronal): {r2_train:.4f}")
    print(f"R2 para el conjunto de validación (Red neuronal): {r2_val:.4f}")
    print(f"R2 para el conjunto de prueba (Red neuronal): {r2_test:.4f}")

    # Arbol de decisión

    #Hiperparametros
    '''
    param_grid_tree = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10]}

    cross_val_plot(X_train, y_train, model_type='tree', param_grid=param_grid_tree)
    '''

    tree_reg = DecisionTreeRegressor(max_depth=10)
    tree_reg.fit(X_train, y_train)

    y_pred_train_tree = tree_reg.predict(X_train)
    y_pred_val_tree = tree_reg.predict(val_data)
    y_pred_test_tree = tree_reg.predict(X_test)

    r2_train_tree = r2_score(y_train, y_pred_train_tree)
    r2_val_tree = r2_score(val_targets, y_pred_val_tree)
    r2_test_tree = r2_score(y_test, y_pred_test_tree)

    print(f"R2 Arbol de decisión (Entrenamiento): {r2_train_tree:.4f}")
    print(f"R2 Arbol de decisión (Validación): {r2_val_tree:.4f}")
    print(f"R2 Arbol de decisión (Prueba): {r2_test_tree:.4f}")

    # Random Forest

    #Hiperparametros
    '''
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 7, 10],
        'max_features': [0.2, 0.5, 0.9, 'sqrt'],
        'min_samples_split': [2, 5, 10]}
    
    cross_val_plot(X_train, y_train, model_type='rf', param_grid=param_grid_rf)
    '''

    rnd_reg = RandomForestRegressor(n_estimators=300, max_leaf_nodes=8, n_jobs=-1, random_state=42)
    rnd_reg.fit(X_train, y_train)

    y_pred_train_rf = rnd_reg.predict(X_train)
    y_pred_val_rf = rnd_reg.predict(val_data)
    y_pred_test_rf = rnd_reg.predict(X_test)

    r2_train_rf = r2_score(y_train, y_pred_train_rf)
    r2_val_rf = r2_score(val_targets, y_pred_val_rf)
    r2_test_rf = r2_score(y_test, y_pred_test_rf)

    print(f"R2 Random Forest (Entrenamiento): {r2_train_rf:.4f}")
    print(f"R2 Random Forest (Validación): {r2_val_rf:.4f}")
    print(f"R2 Random Forest (Prueba): {r2_test_rf:.4f}")

main()