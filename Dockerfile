# Use official Python image
FROM python:3.10


# Set work directory
WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
COPY req2.txt .
RUN pip install --no-cache-dir -r req2.txt

# Copy files
COPY . /app
RUN apt-get update && apt-get install -y ffmpeg

# Run the script
CMD ["python3", "external_infer.py"]


# push to docker hub
# docker tag nn_p abdelrahman370/speaker-gender-age-recognition:v1
# docker push abdelrahman370/speaker-gender-age-recognition:v1

# get started with our project
# clone the repo 
# put you test data in the data folder
# docker pull abdelrahman370/speaker-gender-age-recognition:v1
# docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output -it abdelrahman370/speaker-gender-age-recognition:v1
# the output will be in the data folder, time.txt & results.txt