FROM mem0/mem0-api-server:latest

# Upstream image currently misses runtime dependencies required by default config.
RUN pip install --no-cache-dir \
    "psycopg[binary]>=3.2.0" \
    "psycopg2-binary>=2.9.9" \
    "langchain-neo4j>=0.5.0" \
    "rank-bm25>=0.2.2"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
