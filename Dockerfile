FROM python:latest
RUN mkdir myweb/
COPY main.py myweb/main.py
COPY requirements.txt myweb/requirements.txt
WORKDIR /myweb/
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]