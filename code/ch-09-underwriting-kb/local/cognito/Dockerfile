FROM python:3.12-slim
RUN pip install --no-cache-dir cryptography pyyaml
COPY server.py /server.py
EXPOSE 9229
CMD ["python", "/server.py"]
