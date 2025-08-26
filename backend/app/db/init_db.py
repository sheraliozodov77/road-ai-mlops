# ✅ app/db/init_db.py

from app.db.session import engine
from app.db.models import Base, Prediction  # 👈 Explicitly import the model

def init_db():
    print("[DB INIT] Creating tables...")
    Base.metadata.create_all(bind=engine)
    print("[DB INIT] ✅ Done.")
    
if __name__ == "__main__":
    init_db()