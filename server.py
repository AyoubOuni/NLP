from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
import pickle

# Load dataset, embeddings, and model
def load_dataset():
    try:
        dataset = pd.read_excel("Online Retail.xlsx")
        dataset.dropna(subset=["Description", "InvoiceDate", "Quantity", "UnitPrice"], inplace=True)
        dataset = dataset[(dataset['UnitPrice'] != 0) & (dataset['Quantity'] != 0)]
        dataset["TotalPrice"] = dataset["Quantity"] * dataset["UnitPrice"]
        dataset["InvoiceDate"] = pd.to_datetime(dataset["InvoiceDate"])
        return dataset
    except FileNotFoundError:
        print("Error: Online Retail.xlsx not found.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def load_embeddings():
    try:
        with open('prod_embedding_minilm.pkl', 'rb') as fIn:
            stored_data = pickle.load(fIn)
        return stored_data['sentences'], stored_data['embeddings']
    except FileNotFoundError:
        print("Error: Embedding file not found.")
        return None, None
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None, None

def predict_cluster(recency, frequency, monetary_value, kmeans):
    recency_log = np.log(recency)
    frequency_log = np.log(frequency)
    monetary_value_log = np.log(monetary_value)
    cluster = kmeans.predict([[recency_log, frequency_log, monetary_value_log]])[0]
    return cluster

def get_user_behavior(predicted_cluster):
    cluster_behaviors = {
        "0": "You are the user that purchases infrequently and with lower monetary value. You might be a new customer or an occasional buyer.",
        "1": "You are one of the frequent and high-spending users. You are one of the customers who make larger, regular purchases.",
        "2": "You are the user who purchases moderately often and spends moderately. You are likely a regular customer but not as highly engaged or high-spending."
    }
    return cluster_behaviors.get(str(predicted_cluster), "Invalid cluster ID")

# KMeans Clustering
def perform_clustering(data_cluster):
    features = ['Recency', 'Frequency', 'MonetaryValue']
    X = data_cluster[features]
    X_log = np.log(X)  # Apply logarithmic transformation to handle zero values
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_log)
    return kmeans

def calculate_recommendations(merged_data, data_cluster):
    recommendations = {}
    for cluster in data_cluster['Cluster'].unique():
        cluster_data = merged_data[merged_data['Cluster'] == cluster]
        top_articles = (cluster_data.groupby(['StockCode', 'Description', 'UnitPrice'])
                        .size()
                        .reset_index(name='Count')
                        .sort_values(by='Count', ascending=False)
                        .head(10))
        recommendations[cluster] = top_articles
    return recommendations

# Flask server setup
def server():
    app = Flask(__name__)
    CORS(app)

    dataset = load_dataset()  # Load the dataset
    if dataset is None:
        return jsonify({"message": "Dataset not loaded. Please contact the administrator.", "success": False}), 500

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    unique_descriptions = dataset['Description'].unique()
    embeddings = model.encode(unique_descriptions)

    # Save embeddings to file
    output_file = os.path.join(os.getcwd(), 'prod_embedding_minilm.pkl')
    with open(output_file, "wb") as fOut:
        pickle.dump({'sentences': unique_descriptions, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    # Load Customer Data for Clustering
    file_path = 'Customer_Metrics_with_IDs.csv'
    data_cluster = pd.read_csv(file_path)
    kmeans = perform_clustering(data_cluster)

    @app.route("/")
    def home():
        return jsonify({"message": "Server is running."})

    @app.route("/api/check-items", methods=["POST"])
    def check_items():
        if dataset is None:
            return jsonify({"message": "Dataset not loaded. Please contact the administrator.", "success": False}), 500

        # Parse incoming JSON request
        data = request.json
        if not data or "articles" not in data:
            return jsonify({"message": "Invalid request. 'articles' is required.", "success": False}), 400

        articles = data.get("articles")
        if not isinstance(articles, list) or not articles:
            return jsonify({"message": "'articles' must be a non-empty list.", "success": False}), 400

        purchase_datetimes = []
        purchase_dates = []
        total_monetary = 0
        facture_details = []
        my_articles = []
        missing_articles = []

        for article_info in articles:
            if not isinstance(article_info, dict) or "article" not in article_info or "quantity" not in article_info or "datetime" not in article_info:
                return jsonify({"message": "Each article entry must have 'article', 'quantity', and 'datetime' keys.", "success": False}), 400

            my_articles.append(article_info["article"])
            article_name = article_info["article"]
            quantity = float(article_info["quantity"])
            user_datetime = pd.to_datetime(article_info["datetime"])

            purchase_datetimes.append(user_datetime)
            purchase_dates.append(user_datetime.date())

            item_data = dataset[dataset["Description"].str.contains(article_name, case=False, na=False)]
            if item_data.empty:
                missing_articles.append(article_name)
            else:
                unit_price = item_data["UnitPrice"].iloc[0]
                total_monetary += unit_price * quantity
                facture_details.append({
                    "article": article_name,
                    "unit_price": round(unit_price, 2),
                    "quantity": int(quantity),
                    "datetime": user_datetime.strftime("%Y-%m-%d %H:%M:%S")
                })

        # Calculate recency and frequency
        today = datetime.now().date()
        recency = (today - max(purchase_dates)).days if purchase_dates else None
        frequency = len(sorted(set(purchase_datetimes)))

        # Load embeddings and perform semantic search
        stored_sentences, stored_embeddings = load_embeddings()
        if not stored_sentences or not stored_embeddings:
            return jsonify({"message": "Embeddings not loaded.", "success": False}), 500

        query_embedding = model.encode(my_articles)
        results = []
        for query in query_embedding:
            hits = util.semantic_search(query, stored_embeddings, top_k=5)
            for hit in hits[0]:
                results.append(stored_sentences[hit['corpus_id']])

        # Perform cluster prediction
        predicted_cluster = predict_cluster(recency, frequency, total_monetary, kmeans)

        # Calculate recommendations based on the predicted cluster
        recommendations = calculate_recommendations(merged_data, data_cluster)
        top_articles = recommendations.get(predicted_cluster, [])

        # Prepare response
        response = {
            "message": "All articles processed successfully.",
            "success": True,
            "user_metrics": {
                "recency": recency,
                "frequency": frequency,
                "monetary": round(total_monetary, 2),
            },
            "facture_details": facture_details,
            "recommendations": top_articles.to_dict(orient='records'),
            "result_user": get_user_behavior(predicted_cluster)
        }

        if missing_articles:
            response["missing_articles"] = missing_articles

        return jsonify(response)
    return app
