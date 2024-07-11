FROM python:3.11

WORKDIR /models-api

COPY . .

RUN pip install --upgrade pip && \
    pip install \
        scikit-learn \
        tensorflow \
        joblib \
        fastapi \
        uvicorn \
        python-dotenv \
        pydantic \
        matplotlib \
        numpy \
        uuid \
        polars \
        pyarrow \
        pandas \
        xgboost \
    && pip install -e .

EXPOSE 14506

ENTRYPOINT ["/bin/bash", "-c", "uvicorn src.main:app --host 0.0.0.0 --port 14506"]
