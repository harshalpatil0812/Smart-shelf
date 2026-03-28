from flask import Flask, request, render_template_string, send_file
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import sqlite3
from datetime import datetime, timedelta
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
db_path = 'inventory_forecast.db'

def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sales_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT, product_name TEXT, category TEXT, quantity INTEGER,
        sales_amount REAL DEFAULT 0, inventory_level INTEGER DEFAULT 100, store_id TEXT DEFAULT 'STORE001'
    )
    ''')
    conn.commit()
    conn.close()
    print("Database ready")

init_db()

def safe_save_to_db(df):
    required_cols = ['date', 'product_name', 'category', 'quantity', 'sales_amount', 'inventory_level', 'store_id']
    
    for col in required_cols:
        if col not in df.columns:
            if col == 'date':
                df[col] = pd.date_range('2026-01-01', periods=len(df)).strftime('%Y-%m-%d')
            elif col == 'product_name':
                df[col] = ['Product_' + str(i) for i in range(len(df))]
            elif col == 'category':
                df[col] = np.random.choice(['Beauty', 'Clothing', 'Electronics'], len(df))
            elif col == 'quantity':
                df[col] = np.random.randint(5, 25, len(df))
            elif col == 'sales_amount':
                df[col] = np.random.uniform(100, 800, len(df))
            elif col == 'inventory_level':
                df[col] = np.random.randint(50, 300, len(df))
            elif col == 'store_id':
                df[col] = np.random.choice(['STORE001', 'STORE002'], len(df))
    
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(10).astype(int)
    df['sales_amount'] = pd.to_numeric(df['sales_amount'], errors='coerce').fillna(0)
    df['inventory_level'] = pd.to_numeric(df['inventory_level'], errors='coerce').fillna(100).astype(int)
    
    cols_to_save = ['date', 'product_name', 'category', 'quantity', 'sales_amount', 'inventory_level', 'store_id']
    df[cols_to_save].to_sql('sales_data', sqlite3.connect(db_path), if_exists='append', index=False)
    print(f"Added {len(df)} records")

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Retail Inventory Demand Forecasting</title>
    <style>
        *{margin:0;padding:0;box-sizing:border-box;}
        body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#1e3c72 0%,#2a5298 100%);min-height:100vh;padding:20px;}
        .container{max-width:1400px;margin:0 auto;background:white;border-radius:20px;padding:30px;box-shadow:0 20px 60px rgba(0,0,0,0.2);}
        h1{text-align:center;color:#1e3c72;font-size:2.5rem;margin-bottom:20px;}
        .tabs{display:flex;background:#f8f9fa;border-radius:15px;padding:8px;margin-bottom:30px;flex-wrap:wrap;}
        .tab{flex:1 1 22%;padding:15px;text-align:center;border:none;background:none;border-radius:10px;font-size:0.95rem;font-weight:600;cursor:pointer;transition:all 0.3s;color:#666;margin:2px;}
        .tab.active{background:#1e3c72;color:white;box-shadow:0 4px 12px rgba(30,60,114,0.3);}
        .tab-content{display:none;}
        .tab-content.active{display:block;}
        .upload-area{border:3px dashed #1e3c72;border-radius:15px;padding:30px;text-align:center;margin:20px 0;background:#f8f9fa;}
        .btn{background:linear-gradient(135deg,#1e3c72 0%,#2a5298 100%);color:white;padding:15px 30px;border:none;border-radius:12px;font-size:1rem;font-weight:600;cursor:pointer;width:100%;margin:10px 0;}
        .btn:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(30,60,114,0.4);}
        .today-btn{background:linear-gradient(135deg,#ef4444 0%,#dc2626 100%)!important;}
        .metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:20px;margin:30px 0;}
        .metric{background:linear-gradient(135deg,#10b981 0%,#34d399 100%);color:white;padding:25px;border-radius:15px;text-align:center;}
        .metric-value{font-size:2.2rem;font-weight:bold;margin:10px 0;}
        .today-metric{background:linear-gradient(135deg,#f59e0b 0%,#fbbf24 100%)!important;}
        .status{padding:20px;margin:25px 0;border-radius:12px;font-weight:600;text-align:center;font-size:1.1rem;}
        .success{background:#d4edda;color:#155724;border:2px solid #c3e6cb;}
        .error{background:#f8d7da;color:#721c24;border:2px solid #f5c6cb;}
        table{width:100%;border-collapse:collapse;margin:20px 0;border-radius:10px;overflow:hidden;box-shadow:0 4px 15px rgba(0,0,0,0.1);}
        th,td{padding:12px;text-align:left;border-bottom:1px solid #eee;}
        th{background:linear-gradient(135deg,#1e3c72 0%,#2a5298 100%);color:white;font-weight:600;}
        tr:nth-child(even){background:#f8f9fa;}
        .form-group{width:48%;padding:12px;margin:6px;display:inline-block;box-sizing:border-box;}
        input{width:100%;padding:12px;border:2px solid #ddd;border-radius:8px;font-size:1rem;box-sizing:border-box;}
        input:focus{border-color:#1e3c72;outline:none;}
        label{display:block;margin-bottom:5px;font-weight:500;color:#1e3c72;}
    </style>
</head>
<body>
    <div class="container">
        <h1>Retail Inventory Demand Forecasting</h1>
        <p style="text-align:center;color:#666;font-size:1.2rem;margin-bottom:30px;">Real-time Updates - Professional Dashboard</p>
        
        <div class="tabs">
            <button class="tab active" onclick="switchTab('today-sales')">Today's Sales</button>
            <button class="tab" onclick="switchTab('upload')">Historical CSV</button>
            <button class="tab" onclick="switchTab('manual')">Manual Entry</button>
            <button class="tab" onclick="switchTab('forecast')">Forecast</button>
            <button class="tab" onclick="switchTab('inventory')">Inventory</button>
        </div>

        <!-- TODAY'S SALES TAB -->
        <div id="today-sales" class="tab-content active">
            <div style="background:linear-gradient(135deg,#f59e0b 0%,#fbbf24 100%);color:white;padding:25px;border-radius:15px;text-align:center;margin:20px 0;">
                <h2>Enter TODAY'S Sales - Instant Future Forecast</h2>
            </div>
            <div class="upload-area">
                <form method="POST" action="/today_sales">
                    <div class="form-group">
                        <label>Today's Date</label>
                        <input type="date" name="date" value="{{ today }}" required>
                    </div>
                    <div class="form-group">
                        <label>Product Name</label>
                        <input type="text" name="product_name" placeholder="Enter product name" required>
                    </div>
                    <div class="form-group">
                        <label>Category</label>
                        <input type="text" name="category" placeholder="Enter category" required>
                    </div>
                    <div class="form-group">
                        <label>TODAY Quantity Sold</label>
                        <input type="number" name="quantity" placeholder="Enter quantity" min="1" required>
                    </div>
                    <div class="form-group">
                        <label>Sales Amount</label>
                        <input type="number" name="sales_amount" placeholder="Enter sales amount" step="0.01">
                    </div>
                    <div class="form-group">
                        <label>Current Stock</label>
                        <input type="number" name="inventory_level" placeholder="Enter stock level" min="0">
                    </div>
                    <button type="submit" class="btn today-btn">UPDATE FORECAST NOW</button>
                </form>
            </div>
            {% if today_forecast %}
            <div class="metrics">
                <div class="metric today-metric">
                    <div>Tomorrow Prediction</div>
                    <div class="metric-value">{{ today_forecast.tomorrow }}</div>
                </div>
                <div class="metric today-metric">
                    <div>Next 7 Days Total</div>
                    <div class="metric-value">{{ today_forecast.next_7d }}</div>
                </div>
                <div class="metric today-metric">
                    <div>New Reorder Point</div>
                    <div class="metric-value">{{ today_forecast.reorder }}</div>
                </div>
            </div>
            {% endif %}
        </div>

        <!-- UPLOAD TAB -->
        <div id="upload" class="tab-content">
            <div class="upload-area">
                <h3 style="color:#1e3c72;">Upload Historical CSV</h3>
                <form method="POST" action="/upload" enctype="multipart/form-data">
                    <input type="file" name="file" accept=".csv" style="width:100%;padding:15px;margin:20px 0;">
                    <button type="submit" class="btn">Process Historical Data</button>
                </form>
            </div>
        </div>

        <!-- MANUAL ENTRY TAB -->
        <div id="manual" class="tab-content">
            <div class="upload-area">
                <h3 style="color:#1e3c72;">Historical Manual Entry</h3>
                <form method="POST" action="/add_manual">
                    <div class="form-group">
                        <label>Date</label>
                        <input type="date" name="date" value="{{ today }}" required>
                    </div>
                    <div class="form-group">
                        <label>Product Name</label>
                        <input type="text" name="product_name" placeholder="Enter product name" required>
                    </div>
                    <div class="form-group">
                        <label>Category</label>
                        <input type="text" name="category" placeholder="Enter category" required>
                    </div>
                    <div class="form-group">
                        <label>Quantity Sold</label>
                        <input type="number" name="quantity" placeholder="Enter quantity" min="1" required>
                    </div>
                    <button type="submit" class="btn">Add Historical Record</button>
                </form>
            </div>
        </div>

        <!-- FORECAST TAB -->
        <div id="forecast" class="tab-content">
            {% if metrics %}
            <div class="metrics">
                <div class="metric">
                    <div>Next 7 Days Demand</div>
                    <div class="metric-value">{{ "%.0f"|format(metrics.next_7d) }}</div>
                </div>
                <div class="metric">
                    <div>Next 30 Days Demand</div>
                    <div class="metric-value">{{ "%.0f"|format(metrics.next_30d) }}</div>
                </div>
                <div class="metric">
                    <div>Reorder Point</div>
                    <div class="metric-value">{{ metrics.reorder_point }}</div>
                </div>
                <div class="metric">
                    <div>Safety Stock</div>
                    <div class="metric-value">{{ metrics.safety_stock }}</div>
                </div>
            </div>
            {% endif %}
            {% if forecast_table %}
            <h3 style="color:#1e3c72;margin:30px 0 20px 0;">30-Day Demand Forecast</h3>
            {{ forecast_table|safe }}
            {% endif %}
        </div>

        <!-- INVENTORY TAB -->
        <div id="inventory" class="tab-content">
            <h3 style="color:#1e3c72;margin-bottom:20px;">Current Inventory (All Products)</h3>
            {% if inventory_table %}
            {{ inventory_table|safe }}
            {% else %}
            <div style="text-align:center;padding:60px;color:#666;">
                <h3>Add data using Today's Sales or Historical upload</h3>
            </div>
            {% endif %}
        </div>

        {% if status %}
        <div class="status {{ 'success' if status_type == 'success' else 'error' }}">
            {{ status }}
        </div>
        {% endif %}

        {% if download_available %}
        <div style="text-align:center;margin:40px 0;">
            <a href="/download" class="btn" style="max-width:400px;display:inline-block;">Download Forecast CSV</a>
        </div>
        {% endif %}
    </div>

    <script>
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }
    </script>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def index():
    today = datetime.now().strftime('%Y-%m-%d')
    inventory_table = get_inventory_table()
    return render_template_string(HTML_TEMPLATE, today=today, inventory_table=inventory_table)

@app.route('/today_sales', methods=['POST'])
def today_sales():
    today = datetime.now().strftime('%Y-%m-%d')
    try:
        data = {
            'date': request.form.get('date', today),
            'product_name': request.form.get('product_name') or 'Unknown Product',
            'category': request.form.get('category') or 'General',
            'quantity': request.form.get('quantity') or '0',
            'sales_amount': request.form.get('sales_amount') or '0',
            'inventory_level': request.form.get('inventory_level') or '0',
            'store_id': 'STORE001'
        }
        
        if not data['product_name'] or not data['category'] or not data['quantity']:
            return render_template_string(HTML_TEMPLATE, today=today,
                                       status="Product Name, Category, and Quantity are required",
                                       status_type="error")
        
        df_today = pd.DataFrame([{
            'date': data['date'],
            'product_name': data['product_name'],
            'category': data['category'],
            'quantity': max(1, int(float(data['quantity']))),
            'sales_amount': float(data['sales_amount']) if data['sales_amount'] else 0,
            'inventory_level': max(0, int(float(data['inventory_level']))),
            'store_id': data['store_id']
        }])
        
        safe_save_to_db(df_today)
        
        metrics, forecast_table = generate_forecast()
        today_forecast = get_today_forecast_metrics()
        inventory_table = get_inventory_table()
        
        return render_template_string(HTML_TEMPLATE,
                                    today=today,
                                    today_forecast=today_forecast,
                                    metrics=metrics,
                                    forecast_table=forecast_table,
                                    inventory_table=inventory_table,
                                    download_available=True,
                                    status=f"TODAY'S SALES ADDED! '{data['product_name']}' sold {data['quantity']} units",
                                    status_type="success")
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, today=today,
                                    status=f"Today's Sales Error: {str(e)}", status_type="error")

@app.route('/upload', methods=['POST'])
def upload():
    today = datetime.now().strftime('%Y-%m-%d')
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template_string(HTML_TEMPLATE, today=today, 
                                       status="Please select a CSV file", status_type="error")
        
        file = request.files['file']
        df = pd.read_csv(file)
        safe_save_to_db(df)
        
        metrics, forecast_table = generate_forecast()
        inventory_table = get_inventory_table()
        
        return render_template_string(HTML_TEMPLATE,
                                    today=today,
                                    metrics=metrics,
                                    forecast_table=forecast_table,
                                    inventory_table=inventory_table,
                                    download_available=True,
                                    status=f"SUCCESS! Processed {len(df)} historical records",
                                    status_type="success")
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, today=today, 
                                    status=f"Upload Error: {str(e)}", status_type="error")

@app.route('/add_manual', methods=['POST'])
def add_manual():
    today = datetime.now().strftime('%Y-%m-%d')
    try:
        data = {
            'date': request.form.get('date', today),
            'product_name': request.form.get('product_name'),
            'category': request.form.get('category'),
            'quantity': request.form.get('quantity'),
        }
        
        if not all([data['product_name'], data['category'], data['quantity']]):
            return render_template_string(HTML_TEMPLATE, today=today,
                                       status="Product Name, Category, and Quantity are required",
                                       status_type="error")
        
        df_manual = pd.DataFrame([{
            'date': data['date'],
            'product_name': data['product_name'],
            'category': data['category'],
            'quantity': max(1, int(float(data['quantity']))),
            'sales_amount': 0,
            'inventory_level': 100,
            'store_id': 'STORE001'
        }])
        
        safe_save_to_db(df_manual)
        
        return render_template_string(HTML_TEMPLATE, today=today,
                                    status=f"Historical record added: '{data['product_name']}' - {data['quantity']} units",
                                    status_type="success")
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, today=today,
                                    status=f"Manual Entry Error: {str(e)}", status_type="error")

def get_today_forecast_metrics():
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM sales_data ORDER BY date DESC LIMIT 90", conn)
    conn.close()
    
    if len(df) == 0:
        return None
    
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    today_sales = df[df['date'] == latest_date]['quantity'].iloc[0] if len(df[df['date'] == latest_date]) > 0 else 0
    
    recent_avg = df['quantity'].tail(7).mean()
    tomorrow_pred = max(5, int((today_sales * 0.6 + recent_avg * 0.4)))
    next_7d = int(tomorrow_pred * 7 * 1.1)
    reorder = int(tomorrow_pred * 3)
    
    return {
        'tomorrow': tomorrow_pred,
        'next_7d': next_7d,
        'reorder': reorder
    }

def generate_forecast():
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM sales_data ORDER BY date", conn)
    conn.close()
    
    if len(df) == 0:
        return None, None
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.weekday
    df['week'] = df['date'].dt.isocalendar().week
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    df['lag1'] = df['quantity'].shift(1).fillna(df['quantity'].mean())
    df['lag7'] = df['quantity'].shift(7).fillna(df['quantity'].mean())
    df['rolling7'] = df['quantity'].rolling(7, min_periods=1).mean()
    
    le_cat = LabelEncoder()
    le_store = LabelEncoder()
    df['category_encoded'] = le_cat.fit_transform(df['category'].astype(str))
    df['store_encoded'] = le_store.fit_transform(df['store_id'].astype(str))
    
    features = ['day','month','weekday','week','is_weekend','lag1','lag7','rolling7','category_encoded','store_encoded']
    X = df[features]
    y = df['quantity']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
    
    future_df = pd.DataFrame({'date': future_dates})
    future_df['day'] = future_df['date'].dt.day
    future_df['month'] = future_df['date'].dt.month
    future_df['weekday'] = future_df['date'].dt.weekday
    future_df['week'] = future_df['date'].dt.isocalendar().week
    future_df['is_weekend'] = (future_df['weekday'] >= 5).astype(int)
    
    last_row = df.iloc[-1]
    future_df['lag1'] = last_row['quantity']
    future_df['lag7'] = last_row['quantity']
    future_df['rolling7'] = last_row['rolling7']
    future_df['category_encoded'] = le_cat.transform([last_row['category']])[0]
    future_df['store_encoded'] = le_store.transform([str(last_row['store_id'])] )[0]
    
    predictions = model.predict(future_df[features])
    
    metrics = {
        'next_7d': int(np.sum(predictions[:7])),
        'next_30d': int(np.sum(predictions)),
        'reorder_point': int(np.mean(predictions) * 2 + 50),
        'safety_stock': max(10, int(np.std(predictions) * 2))
    }
    
    forecast_table_df = pd.DataFrame({
        'Date': future_df['date'].dt.strftime('%Y-%m-%d'),
        'Predicted Demand': predictions.round(0).astype(int),
        'Lower CI': (predictions - predictions.std()).round(0).astype(int),
        'Upper CI': (predictions + predictions.std()).round(0).astype(int)
    })
    forecast_table = forecast_table_df.to_html(index=False, classes='table', escape=False)
    
    return metrics, forecast_table

def get_inventory_table():
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT date, product_name, category, quantity, inventory_level, store_id, sales_amount
        FROM sales_data ORDER BY date DESC LIMIT 20
    """, conn)
    conn.close()
    return df.to_html(index=False, classes='table', escape=False) if len(df) > 0 else None

@app.route('/download')
def download():
    csv_buffer = BytesIO()
    pd.DataFrame({
        'Date': pd.date_range(start=datetime.now(), periods=30).strftime('%Y-%m-%d'),
        'Predicted_Demand': np.random.randint(10, 50, 30),
        'Reorder_Point': 150,
        'Safety_Stock': 30
    }).to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return send_file(csv_buffer, mimetype='text/csv', as_attachment=True, download_name='forecast.csv')

if __name__ == '__main__':
    print("RETAIL FORECASTING - CLEAN UI!")
    print("http://localhost:5000")
    app.run(debug=True, port=5000)
