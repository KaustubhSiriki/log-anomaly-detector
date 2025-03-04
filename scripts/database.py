import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")  # Ensure this is set in the .env file

# Initialize database engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class LogEntry(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    event = Column(String, index=True)
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    disk_io = Column(Float)
    network_traffic = Column(Float)
    response_time = Column(Float)
    risk_score = Column(Integer)

# Create tables in the database
Base.metadata.create_all(bind=engine)
