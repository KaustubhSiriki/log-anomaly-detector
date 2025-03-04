from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .database import SessionLocal, LogEntry
from pydantic import BaseModel
from datetime import datetime
from scripts.database import Base, SessionLocal, LogEntry

router = APIRouter()

class LogInput(BaseModel):
    source: str  # The application/server generating the log
    log_level: str  # INFO, ERROR, WARNING, etc.
    message: str  # The log message

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/log/")
def ingest_log(log: LogInput, db: Session = Depends(get_db)):
    db_log = LogEntry(source=log.source, log_level=log.log_level, message=log.message, timestamp=datetime.utcnow())
    db.add(db_log)
    db.commit()
    return {"message": "Log saved successfully"}
