# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir jupyter

# Create a directory for notebooks
RUN mkdir /notebooks

# Copy any existing notebooks into the /notebooks folder in the container
COPY ./notebooks /notebooks

# Expose port 8888 for Jupyter Notebook access
EXPOSE 8888

# Set the command to start Jupyter Notebook and run in the /notebooks directory
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/notebooks"]
