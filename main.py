from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import polars as pl
import re
import uvicorn

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

@app.post("/clean/missing", response_model=CleanedResponse)
async def clean_missing_values(req: MissingValueHandlingRequest):
    try:
        # Convert to Polars DataFrame
        df = pl.DataFrame(req.data)

        rows_before = len(df)
        deleted_rows = 0

        for col in req.columns:
            if req.strategy == "delete":
                df = df.filter(pl.col(col).is_not_null() & (pl.col(col) != "") & pl.col(col).is_not_nan())
            else:
                # Convert to numeric where possible
                df = df.with_columns(
                    pl.col(col).cast(pl.Utf8).str.replace_all(r'^\s*$', 'null').cast(pl.Float64, strict=False).alias(col)
                )

                if req.strategy == "mean":
                    fill_val = df[col].mean()
                elif req.strategy == "median":
                    fill_val = df[col].median()
                elif req.strategy == "zero":
                    fill_val = 0.0
                else:
                    raise HTTPException(400, "Invalid strategy")

                df = df.with_columns(
                    pl.col(col).fill_null(fill_val)
                )

        if req.strategy == "delete":
            deleted_rows = rows_before - len(df)

        return CleanedResponse(
            data=df.to_dicts(),
            message=f"Successfully cleaned {len(req.columns)} columns using {req.strategy}",
            rows_deleted=deleted_rows
        )

    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")


@app.post("/split/column", response_model=CleanedResponse)
async def split_column(req: SplitColumnRequest):
    try:
        df = pl.DataFrame(req.data)
        col = req.column

        # Regex split: extract non-digit prefix and numeric suffix
        prefix_col = f"{col} (Currency)"
        number_col = f"{col} (Amount)"

        # Define extraction
        df = df.with_columns([
            pl.col(col).str.extract(r'^(\D*)', 1).alias(prefix_col),
            pl.col(col).str.extract(r'(\d+\.?\d*)$', 1).cast(pl.Float64, strict=False).alias(number_col)
        ])

        return CleanedResponse(
            data=df.to_dicts(),
            message=f"Successfully split column '{col}' into '{prefix_col}' and '{number_col}'"
        )

    except Exception as e:
        raise HTTPException(500, f"Split failed: {str(e)}")


# === HEALTH CHECK ===
@app.get("/health")
def health():
    return {"status": "OK", "polars_version": pl.__version__}