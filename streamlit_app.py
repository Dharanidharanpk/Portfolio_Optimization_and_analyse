import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize


# Portfolio performance function
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(weights * mean_returns) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std_dev


# Optimization objective: minimize risk for a given return
def minimize_risk(weights, mean_returns, cov_matrix, target_return):
    returns, std_dev = portfolio_performance(weights, mean_returns, cov_matrix)
    return std_dev if returns >= target_return else 1e10  # Penalty if return < target


# Scenario-based portfolio optimization function
def optimize_for_target_return(mean_returns, cov_matrix, target_return):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, target_return)

    # Constraints: weights sum to 1 and return is >= target return
    constraints = (
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
        {'type': 'ineq',
         'fun': lambda weights: portfolio_performance(weights, mean_returns, cov_matrix)[0] - target_return}
    )

    # Bounds: Weights between 0 and 1
    bounds = tuple((0, 1) for asset in range(num_assets))

    # Random initialization for weights
    initial_guess = num_assets * [1. / num_assets, ]

    # Minimize risk while maintaining the target return
    result = minimize(minimize_risk, initial_guess, args=(mean_returns, cov_matrix, target_return),
                      method='SLSQP', bounds=bounds, constraints=constraints)

    return result


# Monte Carlo simulation for portfolio optimization
def monte_carlo_simulation(mean_returns, cov_matrix, num_portfolios=5000, risk_free_rate=0.0175):
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        # Generate random portfolio weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)

        # Calculate portfolio performance
        portfolio_return, portfolio_stddev = portfolio_performance(weights, mean_returns, cov_matrix)

        # Store the results
        results[0, i] = portfolio_return  # Expected return
        results[1, i] = portfolio_stddev  # Risk
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_stddev  # Sharpe ratio

    return results, weights_record


# Predict future stock prices using linear regression
def predict_stock_prices(df, days=90):
    df['Date'] = pd.to_datetime(df.index)
    df['Date'] = df['Date'].map(pd.Timestamp.toordinal)

    X = df['Date'].values.reshape(-1, 1)
    y = df['Adj Close'].values

    model = LinearRegression()
    model.fit(X, y)

    # Predicting future prices
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days)
    future_dates_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    predicted_prices = model.predict(future_dates_ordinal)

    return future_dates, predicted_prices



# Streamlit layout
st.markdown("<h1 style='text-align: center;'>Portfolio Optimization and Analyser</h1>", unsafe_allow_html=True)

# Input: Number of stocks and stock symbols
num_stocks = st.number_input("Number of Stocks in Portfolio", min_value=1, max_value=10, step=1, value=5)
stocks = []
for i in range(num_stocks):
    stock_symbol = st.text_input(f"Enter Stock {i + 1} Symbol", placeholder="AAPL, MSFT, etc.")
    stocks.append(stock_symbol)

# Input: Expected portfolio return
target_return = st.number_input("Expected Portfolio Return (%)", min_value=0.0, value=10.0) / 100

# Input: Market index symbol (for correlation matrix)
market_index = st.text_input("Enter Market Index Symbol", placeholder="^GSPC (S&P 500)")

# Input: Start and End Date for historical data
start_date = st.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
end_date = st.date_input("End Date", value=pd.to_datetime('today'))

# Add an input field for the number of prediction days
prediction_days = st.number_input(
    "Enter Number of Days for Prediction",
    min_value=30,  # Minimum 30 days
    max_value=180,  # Maximum 180 days
    value=90,  # Default value
    step=1,
    help="Enter the number of days you want to predict for each stock."
)

# Fetch stock data
if st.button("Optimize Portfolio"):
    try:
        stock_data = {}
        for stock in stocks:
            stock_data[stock] = yf.download(stock, start=start_date, end=end_date)['Adj Close']

        stock_df = pd.DataFrame(stock_data)
        stock_df1 = pd.DataFrame(stock_data)

        # Fetch market index data
        index_data = yf.download(market_index, start=start_date, end=end_date)['Adj Close']

        stock_df1[market_index] = index_data

        # Calculate daily returns for stocks only (excluding market index)
        daily_returns = stock_df.pct_change().dropna()  # Only for stocks, not the market index

        daily_returns1 = stock_df1.pct_change().dropna()

        # Calculate mean returns and covariance matrix
        mean_returns = daily_returns.mean()
        cov_matrix = daily_returns.cov()

        # Historical stock price trends (using Plotly)
        st.subheader("Historical Stock Price Trends")
        for stock in stocks:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_data[stock].index, y=stock_data[stock], mode='lines', name=stock))
            fig.update_layout(title=f'{stock} Price Trend', xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(fig)

        # Stock Price Comparison Chart (using Plotly)
        st.subheader("Stock Price Comparison Chart")
        fig = go.Figure()
        for stock in stocks:
            fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df[stock], mode='lines', name=stock))
        fig.update_layout(title='Stock Price Comparison', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)


        # Portfolio cumulative returns
        st.subheader("Portfolio Cumulative Returns")
        cumulative_returns = (daily_returns + 1).cumprod()
        st.line_chart(cumulative_returns)

        # Stock correlation heatmap (using Plotly)
        st.subheader("Stock Correlation Heatmap")
        correlation_matrix = daily_returns.corr()
        fig = px.imshow(correlation_matrix, text_auto=True, title="Stock Correlation Matrix", aspect="auto")
        st.plotly_chart(fig)

        # Correlation Matrix with Market Index (using Plotly)
        st.subheader("Correlation Matrix with Market Index")
        fig = px.imshow(daily_returns1.corr(), text_auto=True, title=f"Correlation Matrix (including {market_index})",
                        aspect="auto")
        st.plotly_chart(fig)

        # Concerns Table (Display using Plotly)
        st.subheader("Stock Risk and Return")
        concerns_table = pd.DataFrame({
            'Stock': stocks,
            'Average Return': mean_returns[stocks] * 252,
            'Risk (Standard Deviation)': np.sqrt(np.diag(cov_matrix.loc[stocks, stocks])) * np.sqrt(252)
        })
        st.write(concerns_table)



        # Streamlit layout for stock price trends with future predictions
        st.subheader("Stock Price Trends with Future Predictions")
        # Loop through the selected stocks and plot actual vs predicted prices
        for stock in stocks:
            # Perform prediction with the selected number of days
            future_dates, predicted_prices = predict_stock_prices(yf.download(stock, start=start_date, end=end_date),
                                                                  days=prediction_days)

            # Create the figure for actual vs predicted stock prices
            fig = go.Figure()

            # Add trace for actual prices
            fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df[stock], mode='lines', name='Actual'))

            # Add trace for predicted prices
            fig.add_trace(
                go.Scatter(x=future_dates, y=predicted_prices, mode='lines', name='Predicted', line=dict(dash='dash')))

            # Update layout
            fig.update_layout(
                title=f'{stock} Price with Prediction for {prediction_days} Days',
                xaxis_title='Date',
                yaxis_title='Price'
            )

            # Render the plot
            st.plotly_chart(fig)



        # Optimize for the target return
        results = []
        for i in range(1, num_stocks + 1):
            selected_stocks = stocks[:i]
            if len(selected_stocks) > 0:
                selected_mean_returns = mean_returns[selected_stocks]
                selected_cov_matrix = cov_matrix.loc[selected_stocks, selected_stocks]

                result = optimize_for_target_return(selected_mean_returns, selected_cov_matrix, target_return)

                if result.success:
                    results.append((selected_stocks, result.x, result))

        # Monte Carlo simulation (using Plotly)
        st.subheader("Efficient Frontier ")
        simulation_results, _ = monte_carlo_simulation(mean_returns, cov_matrix)

        # Plotting the simulation results with Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=simulation_results[1], y=simulation_results[0], mode='markers',
                                 marker=dict(color=simulation_results[2], colorscale='Viridis', size=7),
                                 text=["Sharpe: {:.2f}".format(x) for x in simulation_results[2]]))
        fig.update_layout(title="Efficient Frontier",
                          xaxis_title="Risk (Standard Deviation)", yaxis_title="Return")
        st.plotly_chart(fig)

                # Display the optimal portfolio allocation
        if results:
            optimal_portfolio = max(results, key=lambda x:
            portfolio_performance(x[1], mean_returns[x[0]], cov_matrix.loc[x[0], x[0]])[0])
            optimal_weights = optimal_portfolio[1]
            selected_stocks = optimal_portfolio[0]


            allocation_df = pd.DataFrame({
                'Stock': selected_stocks,
                'Price': [stock_data[stock].iloc[-1] for stock in selected_stocks],  # Latest price for each stock
                'Weight': optimal_weights,
            })


            # Optimal portfolio (minimize volatility)
            def minimize_volatility(weights):
                return portfolio_performance(weights, mean_returns, cov_matrix)[1]


            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(len(mean_returns)))
            result = minimize(minimize_volatility, len(mean_returns) * [1. / len(mean_returns)], method='SLSQP',
                              bounds=bounds, constraints=constraints)

            optimal_weights = result['x']

            # Stock distribution pie chart
            st.subheader("Stock Distribution Based on Optimal Weights")
            fig = px.pie(values=optimal_weights, names=stocks, title="Optimal Portfolio Allocation")
            st.plotly_chart(fig)

            #
            st.subheader("Optimal Portfolio Allocation")
            allocation_df = pd.DataFrame({'Stock': stocks, 'Weightage': optimal_weights * 100})
            st.dataframe(allocation_df)

            # Display the portfolio performance
            expected_return, expected_risk = portfolio_performance(optimal_weights, mean_returns[stocks],
                                                                   cov_matrix.loc[stocks, stocks])
            st.write(f"Expected Portfolio Return: {expected_return * 100:.2f}%")
            st.write(f"Portfolio Risk (Standard Deviation): {expected_risk * 100:.2f}%")        
            st.write(f"Sharpe Ratio: {(expected_return - 0.0175) / expected_risk:.2f}")




    except Exception as e:
        st.error(f"An error occurred: {e}")



# Instructions
st.sidebar.title("Instructions")
st.sidebar.write("""
    - Input the number of stocks you want in your portfolio.
    - Enter the stock symbols (e.g., AAPL, MSFT) for those stocks.
    - Specify your expected portfolio return (in %).
    - Choose a market index symbol (e.g., ^GSPC for S&P 500) for correlation analysis.
    - Set the start and end dates for the historical data.
    - Click the "Optimize Portfolio" button to see the results, including the optimal stock allocation, portfolio performance, stock price predictions, correlation matrix, and Monte Carlo simulation.
""")

# Define the app layout and functionality
def main_app():

    # Check if 'Meet the Team' button is pressed
    if st.sidebar.button("Meet the Team"):
        team_info()
        if st.button("Back to App"):
            st.session_state.page = "home"  # Switch back to the main app page


# Define the Team Info page with 2-2-1 layout and center alignment
def team_info():
    # Team section title
    st.markdown("<h3 style='text-align: center;'>Team Developers</h3>", unsafe_allow_html=True)

    # First row (2 people)
    col1, col2 = st.columns(2)

# Developer 1 in the first column
    developer1_image_url = "https://static.vecteezy.com/system/resources/thumbnails/005/346/410/small_2x/close-up-portrait-of-smiling-handsome-young-caucasian-man-face-looking-at-camera-on-isolated-light-gray-studio-background-photo.jpg"  # Replace with your actual Google Drive file ID
    with col1:
        # Center-align image and text
        st.markdown(f"<div style='text-align: center;'><img src='{developer1_image_url}' width='300'></div>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center;'>DHARANIDHARAN P K</h5>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>23MBA0059</p>", unsafe_allow_html=True)

    # Developer 2 in the second column
    developer2_image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ-ZHqWA3ajb0g2TmGMYzSoRpiR5HqjelAKfw&s"  # Replace with your actual Google Drive file ID
    with col2:
        # Center-align image and text
        st.markdown(f"<div style='text-align: center;'><img src='{developer2_image_url}' width='300'></div>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center;'>DHANUSHYA J</h5>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>23MBA0047</p>", unsafe_allow_html=True)

    # Second row (2 people)
    col3, col4 = st.columns(2)

    # Developer 3 in the first column
    developer3_image_url = "https://static.vecteezy.com/system/resources/thumbnails/005/346/410/small_2x/close-up-portrait-of-smiling-handsome-young-caucasian-man-face-looking-at-camera-on-isolated-light-gray-studio-background-photo.jpg"  # Replace with your actual Google Drive file ID
    with col3:
        # Center-align image and text
        st.markdown(f"<div style='text-align: center;'><img src='{developer3_image_url}' width='300'></div>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center;'>YUVARAJ ANAND</h5>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>23MBA0086</p>", unsafe_allow_html=True)

    # Developer 4 in the second column
    developer4_image_url = "https://static.vecteezy.com/system/resources/thumbnails/005/346/410/small_2x/close-up-portrait-of-smiling-handsome-young-caucasian-man-face-looking-at-camera-on-isolated-light-gray-studio-background-photo.jpg"  # Replace with your actual Google Drive file ID
    with col4:
        # Center-align image and text
        st.markdown(f"<div style='text-align: center;'><img src='{developer4_image_url}' width='300'></div>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center;'>VISHVANTH S</h5>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>23MBA0121</p>", unsafe_allow_html=True)

    # Third row (1 person)
    col5 = st.columns(1)[0]  # Create a single column for the last person

    # Developer 5
    developer5_image_url = "https://static.vecteezy.com/system/resources/thumbnails/005/346/410/small_2x/close-up-portrait-of-smiling-handsome-young-caucasian-man-face-looking-at-camera-on-isolated-light-gray-studio-background-photo.jpg"  # Replace with your actual Google Drive file ID
    with col5:
        # Center-align image and text
        st.markdown(f"<div style='text-align: center;'><img src='{developer5_image_url}' width='300'></div>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center;'>NAVEEN KUMAR S</h5>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>23MBA0089</p>", unsafe_allow_html=True)

# Initialize session state if not already set
if "page" not in st.session_state:
    st.session_state.page = "home"  # Default to the home page
    # Add the 'Back to App' button

# Start the app
main_app()
