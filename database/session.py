from sqlalchemy.orm import scoped_session, sessionmaker
from extensions import db

def get_session():
    engine = db.engine
    return scoped_session(sessionmaker(bind=engine))
