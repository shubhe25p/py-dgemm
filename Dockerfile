FROM python:3.8
WORKDIR /app
COPY . /app
RUN ml python
CMD ["python", "python-dgemm.py","--nsize","2048"]
