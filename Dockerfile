# Use official Python image
FROM python:3.11

# Set work directory
WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy files
COPY . /app

# Run the script
CMD ["python3", "external_infer.py"]
