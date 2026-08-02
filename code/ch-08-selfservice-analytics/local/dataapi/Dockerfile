FROM python:3.12-slim
RUN pip install --no-cache-dir redshift-connector
COPY server.py /server.py
EXPOSE 8002
CMD ["python", "/server.py"]
