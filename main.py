import asyncio
import time
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, Index, text
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
    AGE_GROUP_ENGL = Column(String(1024), index=True)  # Indexed for potential queries
    NATIONALITY_ARAB = Column(String(1024))
    NATIONALITY_ENGL = Column(String(1024), index=True)  # Indexed for filtering
    OBSVALUE_OBSV = Column(Float)
    SEX_ARAB = Column(String(1024))
    SEX_ENGL = Column(String(1024), index=True)  # Indexed for filtering
    YEAR_TIME = Column(Integer, index=True)  # Indexed for faster queries
    
    # Composite index for common query patterns
    __table_args__ = (
        Index('idx_year_nationality', 'YEAR_TIME', 'NATIONALITY_ENGL'),
        Index('idx_year_sex', 'YEAR_TIME', 'SEX_ENGL'),
    )

# Database setup with optimizations
DATABASE_URL = "sqlite:///./umrah_stats.db"
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,
        "timeout": 30,
        "isolation_level": None,  # Autocommit mode
    },
    poolclass=StaticPool,  # Keep connections alive
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
        
        # Bulk insert for better performance
        records = []
        for _, row in df.iterrows():
            record = UmrahRecord(
                AGE_GROUP_ARAB=row['AGE_GROUP_ARAB'],
                AGE_GROUP_ENGL=row['AGE_GROUP_ENGL'],
                NATIONALITY_ARAB=row['NATIONALITY_ARAB'],
                NATIONALITY_ENGL=row['NATIONALITY_ENGL'],
                OBSVALUE_OBSV=float(row['OBSVALUE_OBSV']),
                SEX_ARAB=row['SEX_ARAB'],
                SEX_ENGL=row['SEX_ENGL'],
                YEAR_TIME=int(row['YEAR_TIME'])
            )
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
                "YEAR_TIME": r.YEAR_TIME
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
    default_response_class=ORJSONResponse  # Faster JSON serialization
)

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
    year: Optional[int] = None,
    nationality: Optional[str] = None,
    sex: Optional[str] = None,
    age_group: Optional[str] = None
):
    """Get filtered Umrah statistics - uses indexes for fast queries"""
    db = SessionLocal()
    try:
        query = db.query(UmrahRecord)
        
        if year:
            query = query.filter(UmrahRecord.YEAR_TIME == year)
        if nationality:
            query = query.filter(UmrahRecord.NATIONALITY_ENGL == nationality)
        if sex:
            query = query.filter(UmrahRecord.SEX_ENGL == sex)
        if age_group:
            query = query.filter(UmrahRecord.AGE_GROUP_ENGL == age_group)
        
        records = query.all()
        
        return {
            "value": [
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
    
    # try:
    #     import uvloop
    #     config["loop"] = "uvloop"
    #     print("Using uvloop for better performance")
    # except ImportError:
    #     print("uvloop not available, using default event loop")
    
    uvicorn.run("main:app", **config)