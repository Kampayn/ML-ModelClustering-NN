from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
import json
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_DIR = 'models'
REQUIRED_FILES = [
    'kmeans_model.joblib',
    'preprocessor.joblib',
    'pca_model.joblib',
    'hasil_clustering_dengan_label.csv',
    'nn_matching_model.h5',
    'nn_preprocessor.joblib'
]

missing_files = [file for file in REQUIRED_FILES if not os.path.exists(os.path.join(MODEL_DIR, file))]
MODELS_LOADED = not missing_files

if MODELS_LOADED:
    kmeans_model = joblib.load(os.path.join(MODEL_DIR, 'kmeans_model.joblib'))
    preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.joblib'))
    pca_model = joblib.load(os.path.join(MODEL_DIR, 'pca_model.joblib'))
    df_influencer = pd.read_csv(os.path.join(MODEL_DIR, 'hasil_clustering_dengan_label.csv'))
    nn_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'nn_matching_model.h5'))
    nn_preprocessor = joblib.load(os.path.join(MODEL_DIR, 'nn_preprocessor.joblib'))

    # Define allowed values
    allowed_categories = ['food', 'lifestyle', 'comedy', 'beauty', 'fashion']
    allowed_business_info = ['Digital creator', 'Public figure', 'Blogger', 'Artist', 'Entrepreneur']

    # Normalize and correct typos in Category
    df_influencer['Category'] = df_influencer['Category'].fillna('').str.lower().replace({
        'beuaty': 'beauty'
    })

    # Normalize business info capitalization (strip whitespace)
    df_influencer['Business Information'] = df_influencer['Business Information'].fillna('').str.strip()

    # Logging raw unique values
    logging.info(f"Kategori unik sebelum filter: {df_influencer['Category'].unique()}")
    logging.info(f"Business Info unik sebelum filter: {df_influencer['Business Information'].unique()}")

    # Filter dataset
    df_filtered = df_influencer[
        df_influencer['Category'].isin(allowed_categories) &
        df_influencer['Business Information'].isin(allowed_business_info)
    ].copy()

    logging.info(f"Jumlah data setelah filter: {len(df_filtered)}")
    logging.info(f"Kategori hasil filter: {df_filtered['Category'].unique()}")
    logging.info(f"Business Info hasil filter: {df_filtered['Business Information'].unique()}")

else:
    df_filtered = pd.DataFrame()

@app.route('/')
def home():
    return jsonify({
        "status": "ok",
        "models_loaded": MODELS_LOADED,
        "missing_files": missing_files
    })

@app.route('/categories', methods=['GET'])
def get_categories():
    if MODELS_LOADED:
        return jsonify({"mapped_categories": sorted(df_filtered['Category'].unique().tolist())})
    return jsonify({"error": "Models not loaded"}), 503

@app.route('/business-info', methods=['GET'])
def get_business_info():
    if MODELS_LOADED:
        return jsonify({"categories": sorted(df_filtered['Business Information'].unique().tolist())})
    return jsonify({"error": "Models not loaded"}), 503

@app.route('/filtered-data', methods=['GET'])
def get_filtered_data():
    if MODELS_LOADED:
        return jsonify(df_filtered.to_dict(orient='records'))
    return jsonify({"error": "Models not loaded"}), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
