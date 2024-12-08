# Use the official Python 3.9 slim-buster image as the base
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /NLP

# # Copy requirements file to the container
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the application code into the container
# COPY . .

# # Expose the port Flask will run on (Back4App uses PORT environment variable)
# EXPOSE 8080

# # Set environment variables for Flask
# ENV FLASK_APP=app.py
# ENV FLASK_RUN_HOST=0.0.0.0
# ENV FLASK_RUN_PORT=8080

# # Start the Flask application
# CMD ["flask", "run"]

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt requirements.txt


RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD exec gunicorn --workers=1 --threads=4 --timeout=0 server:app
