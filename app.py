from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from statsmodels.tsa.holtwinters import ExponentialSmoothing

app = Flask(__name__)


# Load and prepare the data
def load_data():
    data = pd.read_excel('data/storeData.xls', engine='xlrd')
    data['Order Date'] = pd.to_datetime(data['Order Date'])
    return data


@app.route('/')
def dashboard():
    data = load_data()
    categories = data['Category'].unique()
    return render_template('dashboard.html', categories=categories)


@app.route('/subcategory', methods=['POST'])
def subcategory():
    selected_category = request.form.get('category')
    data = load_data()
    subcategories = data[data['Category'] == selected_category]['Sub-Category'].unique()
    return render_template('subcategory_selection.html', category=selected_category, subcategories=subcategories)


@app.route('/product_selection', methods=['POST'])
def product_selection():
    selected_category = request.form.get('category')
    selected_subcategory = request.form.get('subcategory')
    data = load_data()
    products = data[(data['Category'] == selected_category) & (data['Sub-Category'] == selected_subcategory)][
        'Product Name'].unique()
    return render_template('product_selection.html', category=selected_category, subcategory=selected_subcategory,
                           products=products)


@app.route('/product/<product_name>')
def product_detail(product_name):
    data = load_data()
    product_data = data[data['Product Name'] == product_name]

    # Sales over time plot
    fig, ax = plt.subplots(figsize=(12, 6))  # Increased figure size
    ax.plot(product_data['Order Date'], product_data['Sales'], marker='o', linestyle='-', color='blue', label='Sales')

    # Enhancements
    ax.set_title(f'Sales Over Time for {product_name}', fontsize=16)
    ax.set_xlabel('Order Date', fontsize=14)
    ax.set_ylabel('Sales', fontsize=14)
    ax.grid(True)  # Add grid
    ax.legend()  # Show legend

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45)

    # Convert plot to PNG image
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')  # Use bbox_inches for tight layout
    img.seek(0)
    product_plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Forecast sales using Holt-Winters method for 5 years (60 months)
    product_data.set_index('Order Date', inplace=True)
    product_sales_ts = product_data['Sales'].resample('M').sum()

    # Fit the model with seasonal components
    model = ExponentialSmoothing(product_sales_ts, seasonal='add', seasonal_periods=12)
    model_fit = model.fit()

    # Forecasting for the next 60 months (5 years)
    forecast = model_fit.forecast(60)

    # Plot forecast
    fig, ax = plt.subplots(figsize=(12, 6))  # Increased figure size for forecast
    product_sales_ts.plot(ax=ax, label='Historical Sales', color='blue')
    forecast.plot(ax=ax, label='Forecasted Sales', color='orange')

    # Enhancements for forecast plot
    ax.set_title(f'Sales Forecast for {product_name} (Next 5 Years)', fontsize=16)
    ax.set_xlabel('Order Date', fontsize=14)
    ax.set_ylabel('Sales', fontsize=14)
    ax.grid(True)  # Add grid
    ax.legend()  # Show legend

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45)

    # Convert forecast plot to PNG
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')  # Use bbox_inches for tight layout
    img.seek(0)
    forecast_plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Display relevant product details
    total_sales = product_data['Sales'].sum()
    total_profit = product_data['Profit'].sum()
    total_quantity = product_data['Quantity'].sum()

    return render_template('product_detail.html',
                           product_name=product_name,
                           product_plot_url=product_plot_url,
                           forecast_plot_url=forecast_plot_url,
                           total_sales=total_sales,
                           total_profit=total_profit,
                           total_quantity=total_quantity)


if __name__ == '__main__':
    app.run(debug=True)