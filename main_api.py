
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import text
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np

# Database setup
DATABASE_URL = "sqlite:///./mechanics_db.sqlite"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
metadata = MetaData()

# Staging Table
Ingestion_Stage = Table('Ingestion_Stage', metadata,
    Column('mech_name', String),
    Column('mech_phone', String),
    Column('location_info', String),
    Column('operating_hours', String),
    Column('slot_details', String),
    Column('part_details', String),
    Column('employee_details', String),
    Column('project_info', String),
    Column('client_info', String),
    Column('order_details', String)
)

# BCNF Tables
class Mechanics(Base):
    __tablename__ = 'Mechanics'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    phone_number = Column(String)
    location_id = Column(Integer)

class Locations(Base):
    __tablename__ = 'Locations'
    id = Column(Integer, primary_key=True, index=True)
    address = Column(String)
    operating_hours = Column(String)

class Parts(Base):
    __tablename__ = 'Parts'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)

class Slots(Base):
    __tablename__ = 'Slots'
    id = Column(Integer, primary_key=True, index=True)
    mechanic_id = Column(Integer)
    time_slot = Column(String)

# Predictive Data Table
class Predictive_Data(Base):
    __tablename__ = 'Predictive_Data'
    id = Column(Integer, primary_key=True, index=True)
    Vehicle_Model = Column(String)
    Mileage = Column(Integer)
    Maintenance_History = Column(String)
    Reported_Issues = Column(Integer)
    Vehicle_Age = Column(Integer)
    Need_Maintenance = Column(Integer)

metadata.create_all(bind=engine)
Base.metadata.create_all(bind=engine)

app = FastAPI()

# ML Model Training
def train_and_save_model():
    # Dummy data for training
    data = {
        'Vehicle_Model': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
        'Mileage': [10000, 20000, 15000, 30000, 25000, 12000, 28000],
        'Maintenance_History': ['Good', 'Average', 'Good', 'Poor', 'Average', 'Good', 'Poor'],
        'Reported_Issues': [1, 3, 2, 5, 4, 1, 6],
        'Vehicle_Age': [2, 4, 3, 6, 5, 2, 7],
        'Need_Maintenance': [0, 1, 0, 1, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    df.to_sql('Predictive_Data', engine, if_exists='replace', index=False)

    X = df.drop('Need_Maintenance', axis=1)
    y = df['Need_Maintenance']

    numerical_features = ['Mileage', 'Reported_Issues', 'Vehicle_Age']
    categorical_features = ['Vehicle_Model', 'Maintenance_History']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression())])

    pipeline.fit(X, y)
    joblib.dump(pipeline, 'maintenance_model.joblib')

    imputation_values = {
        'numerical': {col: X[col].mean() for col in numerical_features},
        'categorical': {col: X[col].mode()[0] for col in categorical_features}
    }
    joblib.dump(imputation_values, 'imputation_values.joblib')

train_and_save_model()

model = joblib.load('maintenance_model.joblib')
imputation_values = joblib.load('imputation_values.joblib')

class PredictionInput(BaseModel):
    Vehicle_Model: str | None = None
    Mileage: int | None = None
    Maintenance_History: str | None = None
    Reported_Issues: int | None = None
    Vehicle_Age: int | None = None

@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        df.to_sql('Ingestion_Stage', engine, if_exists='append', index=False)
        return {"message": "CSV uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/data/{table_name}/")
def get_table_data(table_name: str, page: int = 1, page_size: int = 10):
    try:
        conn = engine.connect()
        query = f"SELECT * FROM {table_name} LIMIT {page_size} OFFSET {(page - 1) * page_size}"
        df = pd.read_sql(query, conn)
        conn.close()
        return json.loads(df.to_json(orient='records'))
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Table not found or error: {e}")

@app.post("/normalize/")
def normalize_data():
    try:
        conn = engine.connect()
        # Clear BCNF tables
        conn.execute(text("DELETE FROM Mechanics"))
        conn.execute(text("DELETE FROM Locations"))
        conn.execute(text("DELETE FROM Parts"))
        conn.execute(text("DELETE FROM Slots"))
        
        df = pd.read_sql("SELECT * FROM Ingestion_Stage", conn)

        # Normalize data (this is a simplified example)
        locations = df[['location_info', 'operating_hours']].drop_duplicates().reset_index(drop=True)
        locations['id'] = locations.index + 1
        locations.rename(columns={'location_info': 'address'}, inplace=True)
        locations.to_sql('Locations', engine, if_exists='append', index=False)

        location_map = {row['address']: row['id'] for index, row in locations.iterrows()}

        mechanics = df[['mech_name', 'mech_phone', 'location_info']].drop_duplicates().reset_index(drop=True)
        mechanics['id'] = mechanics.index + 1
        mechanics['location_id'] = mechanics['location_info'].map(location_map)
        mechanics.rename(columns={'mech_name': 'name', 'mech_phone': 'phone_number'}, inplace=True)
        mechanics[['id', 'name', 'phone_number', 'location_id']].to_sql('Mechanics', engine, if_exists='append', index=False)
        
        conn.close()
        return {"message": "Data normalized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/")
def predict(input_data: PredictionInput):
    try:
        data = input_data.dict()
        
        # Impute missing values
        for col, value in imputation_values['numerical'].items():
            if data[col] is None:
                data[col] = value
        for col, value in imputation_values['categorical'].items():
            if data[col] is None:
                data[col] = value

        df = pd.DataFrame([data])
        prediction = model.predict(df)
        probability = model.predict_proba(df)
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(probability[0][prediction[0]])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tables/")
def get_tables():
    inspector = inspect(engine)
    return {"tables": inspector.get_table_names()}
