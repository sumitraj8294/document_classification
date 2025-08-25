from sqlalchemy import (
    create_engine, Column, Integer, String, Text, ForeignKey, DateTime
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
import datetime
import os

DB_URL = os.getenv("DOCINTEL_DB_URL", "sqlite:///documents.db")

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    doc_type = Column(String)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    content = relationship("Content", back_populates="document", uselist=False)
    metadata = relationship("Metadata", back_populates="document", uselist=False)

class Content(Base):
    __tablename__ = "contents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(Integer, ForeignKey("documents.id"), index=True)
    text = Column(Text)
    summary = Column(Text)
    document = relationship("Document", back_populates="content")

class Metadata(Base):
    __tablename__ = "metadata"
    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(Integer, ForeignKey("documents.id"), index=True)
    category = Column(String, index=True)
    document = relationship("Document", back_populates="metadata")

engine = create_engine(DB_URL, echo=False, future=True)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
