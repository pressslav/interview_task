#Use an official Python runtime as a parent image
FROM python:3.9-slim

#Set the working directory in the container
WORKDIR /app

#Copy the requirements file into the container at /app
COPY ./serve/requirements.txt /app/requirements.txt

#Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

#Copy the 'serve' and 'models' directories into the container
COPY ./serve /app/serve
COPY ./models /app/models

#Create the log file to ensure it exists and has the correct permissions
RUN touch /app/serve/app.log

# Expose port 8000 to allow communication to the API
EXPOSE 8000

#Define the command to run the application
#Use 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "serve.main:app", "--host", "0.0.0.0", "--port", "8000"] 