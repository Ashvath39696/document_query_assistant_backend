# Use official Python base image

FROM python:3.10-slim
 
# Set working directory

WORKDIR /app
 
# Copy dependency file and install packages

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt
 
# Copy rest of the code into the container

COPY . .
 
# Expose the port FastAPI will run on

EXPOSE 8000
 
# Healthcheck for FastAPI (optional)

HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1
 
# Start the FastAPI app with Uvicorn

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
 