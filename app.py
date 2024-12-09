from flask_cors import CORS

from server import server
# Initialize Flask app
app = server()
CORS(app)  # Allow all origins for CORS




if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)
    print("Flask server stopped.")
