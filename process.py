import pandas as pd
from sklearn.cluster import KMeans

# Preprocessing and Saving Processed Data
try:
    # Load dataset
    dataset = pd.read_excel("Online Retail.xlsx")
    print("Dataset loaded successfully.")

    # Preprocess dataset
    dataset.dropna(subset=["Description", "InvoiceDate", "Quantity", "UnitPrice", "CustomerID"], inplace=True)
    dataset = dataset[dataset["Quantity"] > 0]  # Remove invalid quantities
    dataset = dataset[~dataset["InvoiceNo"].str.startswith("C", na=False)]  # Exclude canceled transactions
    dataset["TotalPrice"] = dataset["Quantity"] * dataset["UnitPrice"]
    dataset["InvoiceDate"] = pd.to_datetime(dataset["InvoiceDate"])

    print(f"Dataset preprocessing complete. Total records: {len(dataset)}")

    # Compute RFM metrics
    today = dataset["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = dataset.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (today - x.max()).days,  # Recency
        "InvoiceNo": "count",  # Frequency
        "TotalPrice": "sum"    # Monetary
    }).rename(columns={"InvoiceDate": "Recency", "InvoiceNo": "Frequency", "TotalPrice": "Monetary"})

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm["Cluster"] = kmeans.fit_predict(rfm[["Recency", "Frequency", "Monetary"]])
    print("Clustering complete.")

    # Save processed data and RFM with clusters to CSV
    dataset.to_csv("Processed_Online_Retail.csv", index=False)
    rfm.to_csv("RFM_Clustered.csv", index=True)  # Include CustomerID as index
    print("Processed data and RFM metrics saved to CSV.")

except FileNotFoundError:
    print("Error: Online Retail.xlsx not found. Please ensure the file exists.")
except Exception as e:
    print(f"Error during preprocessing: {e}")
