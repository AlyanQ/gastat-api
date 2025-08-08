import asyncio
import time
from typing import List, Literal, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import ORJSONResponse, HTMLResponse, Response
from fastapi.middleware.gzip import GZipMiddleware
import csv
import io
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, Index, text, desc, asc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import pandas as pd
import uvicorn

# Pydantic models for response
class UmrahStatistic(BaseModel):
    AGE_GROUP_ARAB: str
    AGE_GROUP_ENGL: str
    NATIONALITY_ARAB: str
    NATIONALITY_ENGL: str
    OBSVALUE_OBSV: float
    SEX_ARAB: str
    SEX_ENGL: str
    YEAR_TIME: int

    class Config:
        orm_mode = True

class UmrahResponse(BaseModel):
    value: List[UmrahStatistic]

# SQLAlchemy setup
Base = declarative_base()

class UmrahRecord(Base):
    __tablename__ = "umrah_records"
    
    id = Column(Integer, primary_key=True, index=True)
    AGE_GROUP_ARAB = Column(String(1024))
    AGE_GROUP_ENGL = Column(String(1024), index=True)
    NATIONALITY_ARAB = Column(String(1024))
    NATIONALITY_ENGL = Column(String(1024), index=True)
    OBSVALUE_OBSV = Column(Float)
    SEX_ARAB = Column(String(1024))
    SEX_ENGL = Column(String(1024), index=True)
    YEAR_TIME = Column(Integer, index=True)
    
    # Composite indexes for common query patterns
    __table_args__ = (
        Index('idx_year_nationality', 'YEAR_TIME', 'NATIONALITY_ENGL'),
        Index('idx_year_sex', 'YEAR_TIME', 'SEX_ENGL'),
        Index('idx_year_age', 'YEAR_TIME', 'AGE_GROUP_ENGL'),
    )

# Database setup with optimizations
DATABASE_URL = "sqlite:///./umrah_stats.db"
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,
        "timeout": 30,
        "isolation_level": None,
    },
    poolclass=StaticPool,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Global cache for all records
all_records_cache = None
cache_timestamp = 0
CACHE_TTL = 3600  # 1 hour cache

def init_db():
    """Initialize database and load data from CSV"""
    Base.metadata.create_all(bind=engine)
    
    # Check if data already exists
    db = SessionLocal()
    try:
        count = db.query(UmrahRecord).count()
        if count > 0:
            print(f"Database already contains {count} records")
            return
        
        # Load data from CSV
        print("Loading data from CSV...")
        df = pd.read_csv('umrah_data.csv')
        
        # Check if REGION columns exist in the CSV
        has_region_code = 'REGION_CODE' in df.columns
        has_region_engl = 'REGION_ENGL' in df.columns
        
        # Bulk insert for better performance
        records = []
        for _, row in df.iterrows():
            record_data = {
                'AGE_GROUP_ARAB': row['AGE_GROUP_ARAB'],
                'AGE_GROUP_ENGL': row['AGE_GROUP_ENGL'],
                'NATIONALITY_ARAB': row['NATIONALITY_ARAB'],
                'NATIONALITY_ENGL': row['NATIONALITY_ENGL'],
                'OBSVALUE_OBSV': float(row['OBSVALUE_OBSV']),
                'SEX_ARAB': row['SEX_ARAB'],
                'SEX_ENGL': row['SEX_ENGL'],
                'YEAR_TIME': int(row['YEAR_TIME'])
            }
            
            # Add REGION fields if they exist
            if has_region_code:
                record_data['REGION_CODE'] = row.get('REGION_CODE')
            if has_region_engl:
                record_data['REGION_ENGL'] = row.get('REGION_ENGL')
            
            record = UmrahRecord(**record_data)
            records.append(record)
        
        # Bulk insert
        db.bulk_save_objects(records)
        db.commit()
        print(f"Loaded {len(records)} records into database")
        
        # Optimize SQLite for read performance
        db.execute(text("PRAGMA journal_mode=WAL"))
        db.execute(text("PRAGMA synchronous=NORMAL"))
        db.execute(text("PRAGMA cache_size=10000"))
        db.execute(text("PRAGMA page_size=4096"))
        db.execute(text("ANALYZE"))
        db.commit()
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def get_all_records_cached():
    """Get all records with caching"""
    global all_records_cache, cache_timestamp
    
    current_time = time.time()
    
    # Check if cache is valid
    if all_records_cache is not None and (current_time - cache_timestamp) < CACHE_TTL:
        return all_records_cache
    
    # Fetch from database
    db = SessionLocal()
    try:
        records = db.query(UmrahRecord).all()
        # Convert to dict for faster serialization
        all_records_cache = [
            {
                "AGE_GROUP_ARAB": r.AGE_GROUP_ARAB,
                "AGE_GROUP_ENGL": r.AGE_GROUP_ENGL,
                "NATIONALITY_ARAB": r.NATIONALITY_ARAB,
                "NATIONALITY_ENGL": r.NATIONALITY_ENGL,
                "OBSVALUE_OBSV": r.OBSVALUE_OBSV,
                "SEX_ARAB": r.SEX_ARAB,
                "SEX_ENGL": r.SEX_ENGL,
                "YEAR_TIME": r.YEAR_TIME,
                "REGION_CODE": r.REGION_CODE if hasattr(r, 'REGION_CODE') else None,
                "REGION_ENGL": r.REGION_ENGL if hasattr(r, 'REGION_ENGL') else None
            }
            for r in records
        ]
        cache_timestamp = current_time
        return all_records_cache
    finally:
        db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    # Preload cache
    get_all_records_cached()
    yield
    # Shutdown
    pass

# FastAPI app with optimizations
app = FastAPI(
    title="Optimized Umrah Statistics API",
    lifespan=lifespan,
    default_response_class=ORJSONResponse
)

app.add_middleware(GZipMiddleware, minimum_size=500)

@app.get("/v1/stats/DPV_UMRA_PLMUMRA0101_02", response_model=UmrahResponse)
async def get_umrah_statistics():
    """Get all Umrah statistics - optimized endpoint"""
    try:
        # Get cached data
        records = get_all_records_cached()
        return {"value": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/stats/DPV_UMRA_PLMUMRA0101_02/filtered")
async def get_filtered_statistics(
    # Dimension filter (dropdown in UI)
    dimensions: Optional[List[str]] = Query(None, description="Filter dimensions (AGE_GROUP, NATIONALITY)"),
    
    # Individual field filters
    AGE_GROUP_ENGL: Optional[str] = Query(None, description="Filter by age group"),
    NATIONALITY_ENGL: Optional[str] = Query(None, description="Filter by nationality"),
    SEX_ENGL: Optional[str] = Query(None, description="Filter by sex"),
    YEAR_TIME: Optional[int] = Query(None, description="Filter by year"),
    
    # Ordering and pagination
    orderby: Optional[str] = Query(None, description="Order results by field name, prefix with '-' for descending"),
    top: Optional[int] = Query(None, alias="$top", description="Number of results to return"),
    skip: Optional[int] = Query(None, alias="$skip", description="Number of results to skip"),
    
    # Format
    format: Optional[Literal["JSON", "HTML", "CSV"]] = Query("JSON", description="Response format (JSON, HTML, CSV)")
):
    """
    Get filtered Umrah statistics with support for all query parameters shown in the UI.
    
    Query parameters match the OData-style filtering interface:
    - dimensions: Filter by dimension type
    - AGE_GROUP_ENGL, NATIONALITY_ENGL, etc.: Filter by specific field values
    - $orderby: Sort results (e.g., 'YEAR_TIME' for ascending, '-YEAR_TIME' for descending)
    - $top: Limit number of results
    - $skip: Skip first N results (for pagination)
    - format: Response format (JSON, HTML, or CSV)
    """
    db = SessionLocal()
    try:
        query = db.query(UmrahRecord)
        
        # Apply filters based on query parameters
        if AGE_GROUP_ENGL:
            query = query.filter(UmrahRecord.AGE_GROUP_ENGL == AGE_GROUP_ENGL)
        
        if NATIONALITY_ENGL:
            query = query.filter(UmrahRecord.NATIONALITY_ENGL == NATIONALITY_ENGL)
        
        if SEX_ENGL:
            query = query.filter(UmrahRecord.SEX_ENGL == SEX_ENGL)
        
        if YEAR_TIME:
            query = query.filter(UmrahRecord.YEAR_TIME == YEAR_TIME)
        
        # Apply ordering
        if orderby:
            # Check if descending (starts with -)
            if orderby.startswith('-'):
                field_name = orderby[1:]
                order_func = desc
            else:
                field_name = orderby
                order_func = asc
            
            # Map field name to column
            column_map = {
                'AGE_GROUP_ARAB': UmrahRecord.AGE_GROUP_ARAB,
                'AGE_GROUP_ENGL': UmrahRecord.AGE_GROUP_ENGL,
                'NATIONALITY_ARAB': UmrahRecord.NATIONALITY_ARAB,
                'NATIONALITY_ENGL': UmrahRecord.NATIONALITY_ENGL,
                'OBSVALUE_OBSV': UmrahRecord.OBSVALUE_OBSV,
                'SEX_ARAB': UmrahRecord.SEX_ARAB,
                'SEX_ENGL': UmrahRecord.SEX_ENGL,
                'YEAR_TIME': UmrahRecord.YEAR_TIME,
            }
            
            if field_name in column_map:
                query = query.order_by(order_func(column_map[field_name]))
        
        # Apply pagination
        if skip:
            query = query.offset(skip)
        
        if top:
            query = query.limit(top)
        
        # Execute query
        records = query.all()
        
        # Prepare data
        data_list = [
            {
                "AGE_GROUP_ARAB": r.AGE_GROUP_ARAB,
                "AGE_GROUP_ENGL": r.AGE_GROUP_ENGL,
                "NATIONALITY_ARAB": r.NATIONALITY_ARAB,
                "NATIONALITY_ENGL": r.NATIONALITY_ENGL,
                "OBSVALUE_OBSV": r.OBSVALUE_OBSV,
                "SEX_ARAB": r.SEX_ARAB,
                "SEX_ENGL": r.SEX_ENGL,
                "YEAR_TIME": r.YEAR_TIME
            }
            for r in records
        ]
        
        # Return based on format parameter
        format_upper = format.upper() if format else 'JSON'
        
        if format_upper == 'JSON':
            return {"value": data_list}
        
        elif format_upper == 'HTML':
            # Generate simple HTML table
            html_content = '<table border="1"><thead><tr>'
            html_content += '<th>AGE_GROUP_ARAB</th>'
            html_content += '<th>AGE_GROUP_ENGL</th>'
            html_content += '<th>NATIONALITY_ARAB</th>'
            html_content += '<th>NATIONALITY_ENGL</th>'
            html_content += '<th>OBSVALUE_OBSV</th>'
            html_content += '<th>SEX_ARAB</th>'
            html_content += '<th>SEX_ENGL</th>'
            html_content += '<th>YEAR_TIME</th>'
            html_content += '</tr></thead><tbody>'
            
            for record in data_list:
                html_content += '<tr>'
                html_content += f'<td>{record["AGE_GROUP_ARAB"]}</td>'
                html_content += f'<td>{record["AGE_GROUP_ENGL"]}</td>'
                html_content += f'<td>{record["NATIONALITY_ARAB"]}</td>'
                html_content += f'<td>{record["NATIONALITY_ENGL"]}</td>'
                html_content += f'<td>{record["OBSVALUE_OBSV"]}</td>'
                html_content += f'<td>{record["SEX_ARAB"]}</td>'
                html_content += f'<td>{record["SEX_ENGL"]}</td>'
                html_content += f'<td>{record["YEAR_TIME"]}</td>'
                html_content += '</tr>'
            
            html_content += '</tbody></table>'
            
            return HTMLResponse(content=html_content)
        
        elif format_upper == 'CSV':
            # Generate CSV
            output = io.StringIO()
            fieldnames = [
                'AGE_GROUP_ARAB', 'AGE_GROUP_ENGL', 
                'NATIONALITY_ARAB', 'NATIONALITY_ENGL',
                'OBSVALUE_OBSV', 'SEX_ARAB', 'SEX_ENGL', 'YEAR_TIME'
            ]
            
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in data_list:
                writer.writerow(record)
            
            csv_content = output.getvalue()
            
            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=umrah_statistics.csv"}
            )
        
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Format '{format}' not supported. Supported formats are: JSON, HTML, CSV"
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/v1/stats/DPV_UMRA_PLMUMRA0101_02/metadata")
async def get_metadata():
    """Get metadata about available fields and values for filtering"""
    db = SessionLocal()
    try:
        # Get distinct values for each field
        age_groups = db.query(UmrahRecord.AGE_GROUP_ENGL).distinct().all()
        nationalities = db.query(UmrahRecord.NATIONALITY_ENGL).distinct().all()
        sexes = db.query(UmrahRecord.SEX_ENGL).distinct().all()
        years = db.query(UmrahRecord.YEAR_TIME).distinct().order_by(UmrahRecord.YEAR_TIME).all()
        
        # Check if REGION fields exist
        try:
            region_codes = db.query(UmrahRecord.REGION_CODE).distinct().all()
            region_names = db.query(UmrahRecord.REGION_ENGL).distinct().all()
        except:
            region_codes = []
            region_names = []
        
        return {
            "dimensions": ["AGE_GROUP", "NATIONALITY", "REGION"],
            "fields": {
                "AGE_GROUP_ENGL": {
                    "type": "string",
                    "values": sorted([r[0] for r in age_groups if r[0]])
                },
                "NATIONALITY_ENGL": {
                    "type": "string", 
                    "values": sorted([r[0] for r in nationalities if r[0]])
                },
                "SEX_ENGL": {
                    "type": "string",
                    "values": sorted([r[0] for r in sexes if r[0]])
                },
                "YEAR_TIME": {
                    "type": "integer",
                    "values": sorted([r[0] for r in years if r[0]])
                },
                "REGION_CODE": {
                    "type": "string",
                    "values": sorted([r[0] for r in region_codes if r[0]]) if region_codes else []
                },
                "REGION_ENGL": {
                    "type": "string",
                    "values": sorted([r[0] for r in region_names if r[0]]) if region_names else []
                }
            }
        }
    finally:
        db.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    # Configure uvloop only if available (Unix/Linux/macOS)
    config = {
        "host": "0.0.0.0",
        "port": 8002,
        "workers": 1,  # Single worker for SQLite
        "access_log": False  # Disable access logs for performance
    }
    
    try:
        import uvloop
        config["loop"] = "uvloop"
        print("Using uvloop for better performance")
    except ImportError:
        print("uvloop not available, using default event loop")
    
    uvicorn.run("main:app", **config)