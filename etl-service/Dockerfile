FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- CHANGED LINE BELOW ---
# Copy ALL python files (including your new ner_wayang.py)
COPY *.py . 
# --------------------------

COPY ./data ./data

CMD ["python", "./build_graph.py"]