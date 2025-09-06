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

app = FastAPI(title="DataCleaner SaaS Backend", version="1.0")

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

# === ENDPOINTS ===
# === HELPER FUNCTION: SANITIZE DATA FOR JSON SERIALIZATION ===

# === HELPER: SANITIZE OUTPUT FOR JSON ===

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
    """
    Simple health check endpoint to verify the service is running.
    """
    return {
        "status": "OK",
        "service": "DataCleaner SaaS Backend",
        "polars_version": pl.__version__
    }

# === CLEAN MISSING VALUES ENDPOINT ===

@app.post("/clean/missing", response_model=CleanedResponse)
async def clean_missing_values(req: MissingValueHandlingRequest):
    """
    Clean missing values in specified columns using selected strategy.
    Strategies: 'mean', 'median', 'zero', 'delete'
    """
    logger.info(f"▶️ Received /clean/missing request")
    logger.info(f"   Strategy: {req.strategy}")
    logger.info(f"   Columns: {req.columns}")
    logger.info(f"   Input rows: {len(req.data)}")
    if len(req.data) > 0:
        logger.info(f"   Sample input: {req.data[0]}")

    try:
        # Convert to Polars DataFrame
        df = pl.DataFrame(req.data)
        rows_before = len(df)
        deleted_rows = 0

        # Process each selected column
        for col in req.columns:
            # Convert column to string, then to float — handle blanks and non-numeric
            df = df.with_columns(
                pl.col(col)
                .cast(pl.Utf8)  # Ensure string type first
                .str.replace_all(r'^\s*$', 'null')  # Replace blank/whitespace with 'null'
                .cast(pl.Float64, strict=False)  # Convert to float, invalid → null
                .alias(col)
            )

            # Apply cleaning strategy
            if req.strategy == "delete":
                # Remove rows where column is null or NaN
                df = df.filter(pl.col(col).is_not_null() & pl.col(col).is_not_nan())
            else:
                # Compute fill value
                if req.strategy == "mean":
                    fill_val = df[col].mean()
                elif req.strategy == "median":
                    fill_val = df[col].median()
                elif req.strategy == "zero":
                    fill_val = 0.0
                else:
                    raise HTTPException(status_code=400, detail=f"Invalid strategy: {req.strategy}")

                # Fill null values
                df = df.with_columns(
                    pl.col(col).fill_null(fill_val)
                )

        # Count deleted rows if strategy was 'delete'
        if req.strategy == "delete":
            deleted_rows = rows_before - len(df)

        # Sanitize output for JSON serialization
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
    """
    Split a column into two: non-numeric prefix (currency) and numeric suffix (amount)
    Example: 'rs12345' → 'rs' and 12345.0
    """
    logger.info(f"▶️ Received /split/column request")
    logger.info(f"   Column: {req.column}")
    logger.info(f"   Input rows: {len(req.data)}")
    if len(req.data) > 0:
        logger.info(f"   Sample input: {req.data[0]}")

    try:
        # Convert to Polars DataFrame
        df = pl.DataFrame(req.data)
        col = req.column

        # Define new column names
        prefix_col = f"{col} (Currency)"
        number_col = f"{col} (Amount)"

        # Perform split using regex
        df = df.with_columns([
            # Extract non-digit prefix
            pl.col(col)
            .cast(pl.Utf8)
            .str.extract(r'^(\D*)', 1)
            .alias(prefix_col),

            # Extract numeric suffix and convert to float
            pl.col(col)
            .cast(pl.Utf8)
            .str.extract(r'(\d+\.?\d*)$', 1)
            .cast(pl.Float64, strict=False)
            .alias(number_col)
        ])

        # Sanitize output for JSON serialization
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