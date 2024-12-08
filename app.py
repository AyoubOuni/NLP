from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from sklearn.cluster import KMeans
import os
import pickle
from datetime import timedelta

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins for CORS

try:
    # Load dataset
    dataset = pd.read_excel("Online Retail.xlsx")
    print("Dataset loaded successfully.")

    # Preprocess dataset
    dataset.dropna(subset=["Description", "InvoiceDate", "Quantity", "UnitPrice"], inplace=True)
    dataset = dataset[(dataset['UnitPrice'] != 0) & (dataset['Quantity'] != 0)]

########################NLP###################################
    unique_descriptions = dataset['Description'].unique()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(unique_descriptions)
# Define a writable path (e.g., current directory or a specific folder)
    output_file = os.path.join(os.getcwd(), 'prod_embedding_minilm.pkl')

    # Save embeddings using pickle
    with open(output_file, "wb") as fOut:
        print(f"Saving embeddings to {output_file}")
        pickle.dump({'sentences': unique_descriptions, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
##########################################################""
    dataset["TotalPrice"] = dataset["Quantity"] * dataset["UnitPrice"]
    dataset["InvoiceDate"] = pd.to_datetime(dataset["InvoiceDate"])
    print(f"Dataset preprocessing complete. Total records: {len(dataset)}")
    # Convert 'InvoiceDate' to datetime


    dataset['InvoiceDate2'] = pd.to_datetime(dataset['InvoiceDate'], format='%d/%m/%Y %H:%M')

    # Find the most recent date in the dataset
    latest_date = dataset['InvoiceDate2'].max()

    # Get the date 2 weeks before the most recent date
    two_weeks_ago = latest_date - timedelta(weeks=2)

    # Filter the data to include only the last 2 weeks from the latest date
    last_two_weeks_data = dataset[dataset['InvoiceDate'] >= two_weeks_ago]

    # Group by article (Description) and sum the quantity sold
    top_items = last_two_weeks_data.groupby(['Description', 'UnitPrice'])['Quantity'].sum().reset_index()

    # Sort by quantity sold in descending order and take the top 10
    top_10_items = top_items.sort_values(by='Quantity', ascending=False).head(10)

    # Return the top 10 items with their names, quantities, and prices
    top_10_items = top_10_items[['Description', 'UnitPrice', 'Quantity']]
    print('top 10')

    print(top_10_items)
    # Show the top bought articles in the last two weeks

    # Show the top 10 most bought articles in the last two weeks




except FileNotFoundError:
    dataset = None
    print("Error: Online Retail.xlsx not found. Please ensure the file exists.")
except Exception as e:
    dataset = None
    print(f"Error loading dataset: {e}")


@app.route('/')
def hello_world():
    return 'Hello, World!'





# Load the dataset
file_path = 'Customer_Metrics_with_IDs.csv'  # Replace with the path to your CSV file
data_cluster = pd.read_csv(file_path)

# Drop missing values
# data_cluster = data_cluster.dropna()
# data_cluster = data_cluster[(data_cluster['Recency'] != 0) & (data_cluster['Frequency'] != 0) & (data_cluster['MonetaryValue'] != 0)]

# Select features for clustering
features = ['Recency', 'Frequency', 'MonetaryValue']
X = data_cluster[features]

# Apply logarithmic transformation to the features
X_log = np.log(X)  # np.log1p is used to handle zero values safely

# Perform K-Means clustering with k=5
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_log)

# Add cluster labels to the dataset
data_cluster['Cluster'] = kmeans.labels_

# Save the clustered data
data_cluster.to_csv('clustered_data_with_log.csv', index=False)

# Function to predict the cluster for a new data point
def predict_cluster(recency, frequency, monetary_value):
    # Apply logarithmic transformation to the new data point
    recency_log = np.log(recency)
    frequency_log = np.log(frequency)
    monetary_value_log = np.log(monetary_value)
    
    cluster = kmeans.predict([[recency_log, frequency_log, monetary_value_log]])[0]
    return cluster

# Group data by cluster and calculate the mean for each feature
cluster_summary = data_cluster.groupby('Cluster')[['Recency', 'Frequency', 'MonetaryValue']].mean()
cluster_summary['Count'] = data_cluster['Cluster'].value_counts()
print('cluster summary')
print(cluster_summary)



cluster_behaviors = {
    "0": {
        "cluster_id": 0,
        "behavior": "You are the user that purchases infrequently and with lower monetary value. You might be a new customer or an occasional buyer."
    },
    "1": {
        "cluster_id": 1,
        "behavior": "You are one of the frequent and high-spending users. You are one of the customers who make larger, regular purchases."
    },
    "2": {
        "cluster_id": 2,
        "behavior": "You are the user who purchases moderately often and spends moderately. You are likely a regular customer but not as highly engaged or high-spending."
    }
}



def get_user_behavior(predicted_cluster):
    if str(predicted_cluster) in cluster_behaviors:
        return cluster_behaviors[str(predicted_cluster)]["behavior"]
    else:
        return {"error": "Invalid cluster ID"}





transactions = pd.read_excel("Online Retail.xlsx")

#########NLP##################

transactions.dropna(subset=["Description", "InvoiceDate", "Quantity", "UnitPrice"], inplace=True)
transactions = transactions[(transactions['UnitPrice'] != 0) & (transactions['Quantity'] != 0)]

################knn top ###############
data_cluster_path = 'clustered_data.csv'  # Replace with the path to your transactions file
data_cluster = pd.read_csv(data_cluster_path)


# Preprocess transaction data
transactions['CustomerID'] = pd.to_numeric(transactions['CustomerID'], errors='coerce')
transactions = transactions.dropna(subset=['CustomerID'])

# Merge cluster information with transactions
merged_data = pd.merge(transactions, data_cluster[['CustomerID', 'Cluster']], on='CustomerID', how='inner')

def get_top_articles(cluster_data):
        top_articles = (cluster_data.groupby(['StockCode', 'Description','UnitPrice'])
                                .size()
                                .reset_index(name='Count')
                                .sort_values(by='Count', ascending=False)
                                .head(10))
        return top_articles

# Declare recommendations as a global variable
recommendations = {}

# Function to calculate recommendations for each cluster
def calculate_recommendations():
    global recommendations  # Declare it as global to modify it globally
    for cluster in data_cluster['Cluster'].unique():
        cluster_data = merged_data[merged_data['Cluster'] == cluster]
        recommendations[cluster] = get_top_articles(cluster_data)

# Call the function to populate recommendations
calculate_recommendations()

# Print the recommendations
print(recommendations)











@app.route("/api/check-items", methods=["POST"])
def check_items():
    """
    API Endpoint to calculate user-level metrics based on input articles, quantities, and datetime.
    Returns user metrics and invoice details.
    """
    if dataset is None:
        print("Dataset not loaded. Returning error response.")
        return jsonify({"message": "Dataset not loaded. Please contact the administrator.", "success": False}), 500

    # Parse incoming JSON request
    data = request.json
    if not data or "articles" not in data:
        print("Invalid request received: Missing 'articles' key.")
        return jsonify({"message": "Invalid request. 'articles' is required.", "success": False}), 400

    articles = data.get("articles")
    if not isinstance(articles, list) or not articles:
        print("Invalid request received: 'articles' must be a non-empty list.")
        return jsonify({"message": "'articles' must be a non-empty list.", "success": False}), 400

    print(f"Received request for articles: {articles}")

    # Variables to calculate user-level metrics
    purchase_datetimes = []
    purchase_dates = []  # For recency calculation
    total_monetary = 0
    missing_articles = []
    facture_details = []  # To store details of the invoice
    my_articles=[]
    for article_info in articles:
        # Validate the structure of each article
        if not isinstance(article_info, dict) or "article" not in article_info or "quantity" not in article_info or "datetime" not in article_info:
            print(f"Invalid article entry: {article_info}")
            return jsonify({"message": "Each article entry must have 'article', 'quantity', and 'datetime' keys.", "success": False}), 400
        my_articles.append(article_info["article"])
        article_name = article_info["article"]
        quantity = float(article_info["quantity"])
        user_datetime = pd.to_datetime(article_info["datetime"])

        # Collect purchase datetimes for frequency analysis
        purchase_datetimes.append(user_datetime)

        # Collect purchase dates for recency analysis
        purchase_dates.append(user_datetime.date())

        # Filter dataset for the article
        item_data = dataset[dataset["Description"].str.contains(article_name, case=False, na=False)]

        if item_data.empty:
            missing_articles.append(article_name)
        else:
            # Get the unit price from the dataset
            unit_price = item_data["UnitPrice"].iloc[0]  # Take the first matching unit price
            total_monetary += unit_price * quantity  # Add to the monetary total

            # Add article details to the invoice
            facture_details.append({
                "article": article_name,
                "unit_price": round(unit_price, 2),
                "quantity": int(quantity),
                "datetime": user_datetime.strftime("%Y-%m-%d %H:%M:%S")
            })

    # Calculate recency (difference between today and the latest date)
    today = datetime.now().date()  # Use only the date
    if purchase_dates:
        latest_date = max(purchase_dates)
        recency = (today - latest_date).days
    else:
        recency = None  # No valid dates provided

    # Calculate frequency (distinct purchase sessions based on datetime)
    distinct_purchase_datetimes = sorted(set(purchase_datetimes))  # Sort for time difference calculation
    frequency = len(distinct_purchase_datetimes)







    with open('prod_embedding_minilm.pkl', "rb") as fIn:
     stored_data = pickle.load(fIn)

    stored_sentences = stored_data['sentences']

    stored_embeddings = stored_data['embeddings']

    query_embedding = model.encode(my_articles)
    results = []
    for query in query_embedding:
     hits = util.semantic_search(query_embedding, stored_embeddings, top_k=5)
     for hit in hits[0]:
        results.append(stored_sentences[hit['corpus_id']])

    print(results)


    items_with_prices = []

    # Iterate through the results list
    for article_name in results:
        # Find the matching item in the dataset
        item_data = dataset[dataset["Description"].str.contains(article_name, case=False, na=False)]
        
        if item_data.empty:
            # If no matching item is found, add to the missing_articles list
            missing_articles.append(article_name)
        else:
            # If item found, get the first matching price
            unit_price = item_data["UnitPrice"].iloc[0]  # Take the first matching unit price
            # Append the item and its price to the final list
            items_with_prices.append({
                "description": article_name,
                "price": unit_price
            })

    predicted_cluster = predict_cluster(recency, frequency, total_monetary)
    print(f"The metrics Recency={recency}, Frequency={frequency}, MonetaryValue={total_monetary} belong to Cluster {predicted_cluster}.")




        # Display top 10 recommended articles for the predicted cluster
    print(f"\nTop 10 recommended articles for Cluster {predicted_cluster}:")
    top_articles = recommendations.get(predicted_cluster, [])
    print(top_articles)

    # Create a DataFrame
    top_articles_df = pd.DataFrame(top_articles)
    top_articles_df2 = pd.DataFrame(items_with_prices)
    top_articles_df3 = pd.DataFrame(top_10_items)
    print(top_articles_df2)
    # Convert DataFrame to JSON
    top_articles_json = top_articles_df.to_dict(orient='records')
    top_articles_json2 = top_articles_df2.to_dict(orient='records')
    top_articles_json3 = top_articles_df3.to_dict(orient='records')
    print('fff')
    print(top_articles_json2)









    result_user = get_user_behavior(predicted_cluster)

    print(result_user)


    # Prepare response with user-level metrics and invoice details
    response = {
        "message": "All articles processed successfully.",
        "success": True,
        "user_metrics": {
            "recency": recency,
            "frequency": frequency,
            "monetary": round(total_monetary, 2),
        },
        "facture_details": facture_details,
        "recommendations": top_articles_json ,
        "recommendations2": top_articles_json2,
        "recommendations3": top_articles_json3,
        "result_user": result_user,
    }

    if missing_articles:
        response["missing_articles"] = missing_articles

    return jsonify(response)







if __name__ == "__main__":
    print("Starting Flask server...")
    # Get the port from the environment variable, default to 5000 if not set
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(debug=True, host="0.0.0.0", port=5000)
    print("Flask server stopped.")
