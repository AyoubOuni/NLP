# Use the official Python 3.10 slim-buster image as the base
FROM python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /NLP

# Copy the requirements file first to leverage Docker's cache
COPY ./requirements.txt /NLP/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port Flask will run on (Back4App uses PORT environment variable)
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=server.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Use Gunicorn to run the Flask app for better performance
CMD ["gunicorn", "--workers=1", "--threads=4", "--timeout=0", "server:app"]
