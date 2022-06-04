FROM python:3
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . . 
EXPOSE 8050
CMD ["python", "./interactive.py"]