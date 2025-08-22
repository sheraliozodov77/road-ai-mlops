from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, index=True)
    filename = Column(Text)
    input_type = Column(String)
    model_name = Column(String)
    status = Column(String)
    runtime = Column(String, nullable=True)
    created_at = Column(DateTime)
    output_path = Column(Text)
