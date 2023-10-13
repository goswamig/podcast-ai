# Use the official Ubuntu base image
FROM ubuntu:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the entire current directory into the container at /app
COPY . /app

# Install Python
RUN apt-get update && \
    apt-get install -y python3

# Run the hello.py script
CMD ["python3", "hello.py"]

