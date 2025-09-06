import re
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
import polars as pl
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("datacleaner")

app = FastAPI(title="DataCleaner SaaS Backend", version="1.1")

# === DATA MODELS ===

class MissingValueHandlingRequest(BaseModel):
    columns: List[str]  # column names
    strategy: str       # "mean", "median", "zero", "delete"
    data: List[dict]    # [{col1: val, col2: val}, ...]

class SplitColumnRequest(BaseModel):
    column: str         # column name to split
    data: List[dict]    # original data

class CleanedResponse(BaseModel):
    data: List[dict]
    message: str
    rows_deleted: Optional[int] = 0


# === HELPER FUNCTIONS ===

def sanitize_output(data: List[dict]) -> List[dict]:
    """Ensure all values are JSON-serializable"""
    def clean_value(v):
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
        if isinstance(v, (pl.Date, pl.Datetime, pl.Time, pl.Duration)):
            return str(v)
        if v is None:
            return None
        return v
    return [
        {k: clean_value(v) for k, v in row.items()}
        for row in data
    ]


# === HEALTH CHECK ENDPOINT ===

@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "service": "DataCleaner SaaS Backend",
        "polars_version": pl.__version__
    }


# === INSPECT SCHEMA ENDPOINT ===

@app.post("/inspect/schema")
async def inspect_schema(payload: dict):
    """
    Inspect how Polars infers schema from provided data.
    Example body:
    {
      "data": [{"Product": "A", "Price": "100"}, {"Product": "B", "Price": ""}]
    }
    """
    try:
        data = payload.get("data", [])
        if not data:
            raise HTTPException(status_code=400, detail="No data provided")

        df = pl.DataFrame(data, infer_schema_length=0)
        schema_before = df.schema

        # Try casting numeric-looking columns to Float64
        casted_df = df.with_columns([
            pl.when(pl.col(col).str.strip_chars().is_in(["", None]))
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
            for col in df.columns
        ])

        schema_after = casted_df.schema

        return {
            "schema_before": {k: str(v) for k, v in schema_before.items()},
            "schema_after": {k: str(v) for k, v in schema_after.items()},
            "sample_row": casted_df.to_dicts()[0] if len(casted_df) > 0 else {}
        }

    except Exception as e:
        logger.error(f"❌ Error in /inspect/schema: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Schema inspection failed: {str(e)}")


# === CLEAN MISSING VALUES ENDPOINT ===

@app.post("/clean/missing", response_model=CleanedResponse)
async def clean_missing_values(req: MissingValueHandlingRequest):
    logger.info(f"▶️ Received /clean/missing request")
    logger.info(f"   Strategy: {req.strategy}")
    logger.info(f"   Columns: {req.columns}")
    logger.info(f"   Input rows: {len(req.data)}")
    if len(req.data) > 0:
        logger.info(f"   Sample input: {req.data[0]}")

    try:
        df = pl.DataFrame(req.data, infer_schema_length=0)
        logger.info(f"Schema before processing: {df.schema}")
        rows_before = len(df)
        deleted_rows = 0

        for col in req.columns:
            # Normalize blanks → None, then cast to float
            df = df.with_columns(
                pl.col(col)
                .cast(pl.Utf8)
                .str.strip_chars()
                .replace("", None)
                .cast(pl.Float64, strict=False)
                .alias(col)
            )

            # Apply strategy
            if req.strategy == "delete":
                df = df.filter(pl.col(col).is_not_null() & pl.col(col).is_not_nan())
            else:
                if req.strategy == "mean":
                    fill_val = df[col].mean()
                elif req.strategy == "median":
                    fill_val = df[col].median()
                elif req.strategy == "zero":
                    fill_val = 0.0
                else:
                    raise HTTPException(status_code=400, detail=f"Invalid strategy: {req.strategy}")

                df = df.with_columns(
                    pl.col(col).fill_null(fill_val)
                )

        if req.strategy == "delete":
            deleted_rows = rows_before - len(df)

        logger.info(f"Schema after processing: {df.schema}")

        cleaned_data = sanitize_output(df.to_dicts())
        logger.info(f"✅ Processing complete. Output rows: {len(cleaned_data)}")
        if len(cleaned_data) > 0:
            logger.info(f"   Sample output: {cleaned_data[0]}")

        return CleanedResponse(
            data=cleaned_data,
            message=f"Successfully cleaned {len(req.columns)} columns using '{req.strategy}' strategy",
            rows_deleted=deleted_rows
        )

    except Exception as e:
        logger.error(f"❌ Error in /clean/missing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# === SPLIT COLUMN ENDPOINT ===

@app.post("/split/column", response_model=CleanedResponse)
async def split_column(req: SplitColumnRequest):
    logger.info(f"▶️ Received /split/column request")
    logger.info(f"   Column: {req.column}")
    logger.info(f"   Input rows: {len(req.data)}")
    if len(req.data) > 0:
        logger.info(f"   Sample input: {req.data[0]}")

    try:
        df = pl.DataFrame(req.data, infer_schema_length=0)
        col = req.column
        prefix_col = f"{col} (Currency)"
        number_col = f"{col} (Amount)"

        df = df.with_columns([
            pl.col(col)
            .cast(pl.Utf8)
            .str.extract(r'^(\D*)', 1)
            .alias(prefix_col),

            pl.col(col)
            .cast(pl.Utf8)
            .str.extract(r'(\d+\.?\d*)$', 1)
            .cast(pl.Float64, strict=False)
            .alias(number_col)
        ])

        cleaned_data = sanitize_output(df.to_dicts())
        logger.info(f"✅ Processing complete. Output rows: {len(cleaned_data)}")
        if len(cleaned_data) > 0:
            logger.info(f"   Sample output: {cleaned_data[0]}")

        return CleanedResponse(
            data=cleaned_data,
            message=f"Successfully split column '{col}' into '{prefix_col}' and '{number_col}'"
        )

    except Exception as e:
        logger.error(f"❌ Error in /split/column: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Split failed: {str(e)}")


# === ENTRY POINT ===

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
