from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
import json
from werkzeug.exceptions import NotFound
import logging

# Initialize the Flask app and enable logging
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

MODEL_DIR = 'models'
REQUIRED_FILES = [
    'kmeans_model.joblib',
    'preprocessor.joblib',
    'tf_engagement_model.h5',
    'hasil_clustering_dengan_label.csv',
    'model_metadata.json',
    'pca_model.joblib'
]

# Log the required files check
missing_files = []
for file in REQUIRED_FILES:
    file_path = os.path.join(MODEL_DIR, file)
    if not os.path.exists(file_path):
        missing_files.append(file)

if missing_files:
    logging.warning(f"PERINGATAN: File berikut tidak ditemukan: {', '.join(missing_files)}")
    logging.warning("Beberapa fitur API mungkin tidak akan berfungsi.")
else:
    logging.info("Semua file model ditemukan. API siap digunakan.")

try:
    logging.info("Loading KMeans model...")
    kmeans_model = joblib.load(os.path.join(MODEL_DIR, 'kmeans_model.joblib'))
    logging.info("Model KMeans berhasil dimuat.")
    
    logging.info("Loading preprocessor...")
    preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.joblib'))
    logging.info("Preprocessor berhasil dimuat.")
    
    logging.info("Loading TensorFlow model...")
    tf_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'tf_engagement_model.h5'))
    logging.info("Model TensorFlow berhasil dimuat.")
    
    logging.info("Loading PCA model...")
    pca_model = joblib.load(os.path.join(MODEL_DIR, 'pca_model.joblib'))
    logging.info("Model PCA berhasil dimuat.")
    
    logging.info("Loading influencer data...")
    df_influencer = pd.read_csv(os.path.join(MODEL_DIR, 'hasil_clustering_dengan_label.csv'))
    logging.info(f"Data influencer berhasil dimuat. Total: {len(df_influencer)} influencer.")
    
    logging.info("Loading model metadata...")
    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)
    logging.info("Metadata model berhasil dimuat.")
    
    if 'clustering' in metadata and 'cluster_tiers' in metadata['clustering']:
        cluster_tiers = metadata['clustering']['cluster_tiers']
    else:
        cluster_tiers = {}
        for cluster in df_influencer['Cluster'].unique():
            tier = df_influencer[df_influencer['Cluster'] == cluster]['Tier'].iloc[0]
            cluster_tiers[str(cluster)] = tier
    
    logging.info(f"Informasi tier cluster: {cluster_tiers}")
    
    available_categories = df_influencer['Category'].unique().tolist()
    logging.info(f"Kategori yang tersedia: {available_categories}")
    
    available_tiers = df_influencer['Tier'].unique().tolist()
    logging.info(f"Tier yang tersedia: {available_tiers}")
    
    MODELS_LOADED = True
    
except Exception as e:
    logging.error(f"Error saat memuat model: {str(e)}")
    logging.error("API akan berjalan dalam mode terbatas.")
    MODELS_LOADED = False

# Function to preprocess input data
def preprocess_input(data):
    """
    Preprocess input JSON data into a pandas DataFrame for model prediction.
    
    Args:
        data (dict): JSON input data containing fields like 'Followers', 'Engagement Rate', etc.
    
    Returns:
        pd.DataFrame: DataFrame with processed features matching the preprocessor's expectations.
    """
    # Define expected columns based on preprocessor requirements
    expected_columns = [
        'Followers',
        'Engagement Rate',
        'Average Likes',
        'Average Comments',
        'Is Professional Account',
        'Is Verified',
        'Category',
        'Mapped_Label'  # We ensure 'Mapped_Label' is expected, but we don't assign a random label
    ]
    
    # Create a DataFrame from the input dictionary
    input_df = pd.DataFrame([data])
    
    # Ensure all expected columns are present, fill missing ones with appropriate defaults
    for col in expected_columns:
        if col not in input_df.columns:
            if col == 'Mapped_Label':  # If 'Mapped_Label' is missing, handle it without assigning a default
                return {"error": "Missing required column: 'Mapped_Label'"}
            elif col in ['Is Professional Account', 'Is Verified']:
                input_df[col] = False  # Default to False for boolean columns
            elif col == 'Category':
                input_df[col] = 'Unknown'  # Default for categorical columns
            else:
                input_df[col] = 0  # Default to 0 for numerical columns
    
    # Reorder columns to match the preprocessor's expected order
    input_df = input_df[expected_columns]
    
    # Handle missing values in other columns
    input_df = input_df.fillna({
        'Followers': 0,
        'Engagement Rate': 0,
        'Average Likes': 0,
        'Average Comments': 0,
        'Is Professional Account': False,
        'Is Verified': False,
        'Category': 'Unknown',
        'Mapped_Label': 'Unknown'  # If needed, this will be handled later without using a fake label
    })
    
    logging.info(f"Processed input DataFrame: {input_df.to_dict()}")
    return input_df


# Function to recommend influencers
def recommend_influencers(tier=None, business_information=None, top_n=5, sort_by='Engagement Rate'):
    filtered = df_influencer.copy()
    
    # Filter by tier if specified
    if tier:
        filtered = filtered[filtered['Tier'] == tier]
        if filtered.empty:
            return {"message": "No influencers found for the specified tier."}
    
    # Filter by business information (category) if specified
    if business_information:
        filtered = filtered[filtered['Category'] == business_information]
        if filtered.empty:
            return {"message": "No influencers found for the specified category."}
    
    # Sort by engagement rate or specified metric
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(by=sort_by, ascending=False).head(top_n)
    else:
        filtered = filtered.sort_values(by='Engagement Rate', ascending=False).head(top_n)
    
    # Select only the most relevant columns for output
    result = filtered[['Username', 'Category', 'Tier', 'Followers', 'Engagement Rate']]

    # Provide summary of recommendations
    summary = {
        "count": len(result),
        "tier": tier if tier else "All tiers",
        "category": business_information if business_information else "All categories",
        "recommendations": result.to_dict(orient='records')
    }
    
    # Log the output for debugging purposes
    logging.info(f"Recommendations summary: {summary}")
    
    return summary


# Function to predict cluster
def predict_cluster(data):
    input_df = preprocess_input(data)
    input_processed = preprocessor.transform(input_df)
    
    cluster = int(kmeans_model.predict(input_processed)[0])
    tier = cluster_tiers.get(str(cluster), "Undefined")
    
    logging.info(f"Predicted cluster: {cluster}, Predicted tier: {tier}")
    return {
        "cluster": cluster,
        "tier": tier
    }

# Function to predict engagement rate
def predict_engagement(data):
    input_df = preprocess_input(data)
    input_processed = preprocessor.transform(input_df)
    
    if hasattr(input_processed, 'toarray'):
        input_processed = input_processed.toarray()
    
    engagement_prediction = float(tf_model.predict(input_processed)[0][0])
    
    logging.info(f"Predicted engagement rate: {engagement_prediction}")
    return {
        "predicted_engagement_rate": engagement_prediction
    }

# Function for PCA visualization
def visualize_with_pca(data):
    input_df = preprocess_input(data)
    input_processed = preprocessor.transform(input_df)
    
    if hasattr(input_processed, 'toarray'):
        pca_result = pca_model.transform(input_processed.toarray())
    else:
        pca_result = pca_model.transform(input_processed)
    
    logging.info(f"PCA coordinates: {pca_result[0]}")
    return {
        "pca_coordinates": {
            "x": float(pca_result[0][0]),
            "y": float(pca_result[0][1])
        }
    }

@app.route('/')
def index():
    """Root endpoint untuk API"""
    return jsonify({
        "status": "online",
        "message": "Influencer Recommendation API",
        "models_loaded": MODELS_LOADED,
        "endpoints": [
            {"method": "GET", "path": "/health", "description": "Check API health"},
            {"method": "GET", "path": "/metadata", "description": "Get model metadata"},
            {"method": "GET", "path": "/categories", "description": "Get available categories"},
            {"method": "GET", "path": "/tiers", "description": "Get available tiers"},
            {"method": "GET", "path": "/recommend", "description": "Get influencer recommendations"},
            {"method": "POST", "path": "/predict", "description": "Predict cluster, tier and engagement rate"}
        ]
    })

@app.route('/health')
def health_check():
    """Endpoint untuk health check"""
    return jsonify({
        "status": "healthy",
        "models_loaded": MODELS_LOADED,
        "missing_files": missing_files if missing_files else None
    })

@app.route('/metadata')
def get_metadata():
    """Endpoint untuk mendapatkan metadata model"""
    if MODELS_LOADED:
        return jsonify(metadata)
    else:
        return jsonify({"error": "Models not loaded"}), 503

@app.route('/categories')
def get_categories():
    """Endpoint untuk mendapatkan kategori yang tersedia"""
    if MODELS_LOADED:
        return jsonify({"categories": available_categories})
    else:
        return jsonify({"error": "Models not loaded"}), 503

@app.route('/tiers')
def get_tiers():
    """Endpoint untuk mendapatkan tier yang tersedia"""
    if MODELS_LOADED:
        return jsonify({"tiers": available_tiers})
    else:
        return jsonify({"error": "Models not loaded"}), 503

@app.route('/recommend')
def recommend():
    """Endpoint untuk mendapatkan rekomendasi influencer"""
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded"}), 503
    
    try:
        tier = request.args.get('tier')
        business_information = request.args.get('business_information')
        top_n = int(request.args.get('top_n', 5))
        sort_by = request.args.get('sort_by', 'Engagement Rate')
        
        if tier and tier not in available_tiers:
            return jsonify({
                "error": f"Invalid tier: {tier}. Available tiers: {available_tiers}"
            }), 400
        
        if business_information and business_information not in available_categories:
            return jsonify({
                "error": f"Invalid business information: {business_information}. Available categories: {available_categories}"
            }), 400
        
        recommendations = recommend_influencers(tier, business_information, top_n, sort_by)
        
        return jsonify({
            "count": len(recommendations),
            "tier": tier,
            "business_information": business_information,
            "sort_by": sort_by,
            "recommendations": recommendations
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/usernames_and_categories', methods=['GET'])
def get_usernames_and_categories():
    """Endpoint untuk mendapatkan Username dan Kategori dari data influencer"""
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded"}), 503
    
    try:
        usernames_and_categories = df_influencer[['Username', 'Category']].dropna()
        return jsonify(usernames_and_categories.to_dict(orient='records'))
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk memprediksi cluster, tier, dan engagement rate"""
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded"}), 503
    
    try:
        # Ambil data dari request
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validasi data
        required_fields = ['Followers', 'Engagement Rate']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Prediksi cluster dan tier
        cluster_result = predict_cluster(data)
        
        # Prediksi engagement rate
        engagement_result = predict_engagement(data)
        
        # PCA visualization (optional)
        try:
            pca_result = visualize_with_pca(data)
        except:
            pca_result = {"pca_coordinates": None}
        
        # Gabungkan hasil
        result = {
            **data,
            **cluster_result,
            **engagement_result,
            **pca_result
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = 5001  # Use port 5001
    
    logging.info(f"Starting Flask application on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
