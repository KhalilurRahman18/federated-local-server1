import pickle
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def predict(model, data):
    # Assuming data is a dictionary with the required input features
    features = [data[key] for key in sorted(data.keys())]
    prediction = model.predict([features])
    return {'prediction': prediction[0]}

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def preprocess_data(df, training=True):
    le = LabelEncoder()

    for col in ['tipe_transaksi', 'akun_bank_pengirim', 'instrument_transaksi_pengirim', 'akun_bank_tujuan', 'instrument_tujuan']:
        df[col] = le.fit_transform(df[col])

    df.drop(columns=['tanggal_transaksi', 'no_rekening_pengirim', 'nama_pengirim', 'no_rekening_tujuan', 'nama_tujuan'], inplace=True)

    scaler = StandardScaler()
    df[['jumlah_transaksi']] = scaler.fit_transform(df[['jumlah_transaksi']])

    if training:
        X = df.drop(columns=['is_fraud'])
        y = df['is_fraud']
        return df, X, y
    else:
        X = df.drop(columns=['is_fraud'], errors='ignore')
        return df, X, None
