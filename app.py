import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

with open('encoders/Category_encoder.pkl', 'rb') as file:
    category_encoder = pickle.load(file)
    
with open('models/best_demand_model.pkl', 'rb') as file:
    model = pickle.load(file)

class request_model(BaseModel):
    Category: str
    Inventory_Level: int
    Price: float
    Discount: int
    Promotion: int
    Compititor_Price: float
    

@app.post('/predict')
def predict_demand(request: request_model):
    
    encoded_category = category_encoder['Category'].transform([request.Category])[0]
    
    input_data = pd.DataFrame([{
        'Category': encoded_category,
        'Inventory Level': request.Inventory_Level,
        'Price': request.Price,
        'Discount': request.Discount,
        'Promotion': request.Promotion,
        'Competitor Pricing': request.Compititor_Price
    }])
    
    y_pred = model.predict(input_data)
    
    return {'predicted_demand': float(y_pred[0])}
    
    