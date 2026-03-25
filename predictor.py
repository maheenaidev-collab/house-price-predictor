import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data():
    np.random.seed(42)
    n = 500
    
    data = pd.DataFrame({
        'size_sqft': np.random.randint(500, 5000, n),
        'bedrooms': np.random.randint(1, 6, n),
        'bathrooms': np.random.randint(1, 4, n),
        'age_years': np.random.randint(0, 50, n),
    })
    
    data['price'] = (
        data['size_sqft'] * 150 +
        data['bedrooms'] * 20000 +
        data['bathrooms'] * 15000 -
        data['age_years'] * 5000 +
        np.random.randint(-20000, 20000, n)
    )
    
    return data

def train_model(data):
    X = data[['size_sqft', 'bedrooms', 'bathrooms', 'age_years']]
    y = data['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Performance:")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R² Score: {r2:.4f}")
    
    return model

def predict_price(model, size, bedrooms, bathrooms, age):
    input_data = np.array([[size, bedrooms, bathrooms, age]])
    price = model.predict(input_data)[0]
    print(f"\nPredicted Price: ${price:,.2f}")
    return price

def main():
    print("🏠 House Price Predictor\n")
    
    data = load_data()
    print(f"Dataset: {len(data)} houses loaded\n")
    
    model = train_model(data)
    
    print("\n--- Try a Prediction ---")
    predict_price(model, size=2000, bedrooms=3, bathrooms=2, age=10)
    predict_price(model, size=3500, bedrooms=4, bathrooms=3, age=5)

if __name__ == "__main__":
    main()
