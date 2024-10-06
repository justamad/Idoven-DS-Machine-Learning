FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir jupyter
RUN mkdir /notebooks
RUN pip install --no-cache-dir -r requirements.txt

COPY ./notebooks /notebooks

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/notebooks"]
