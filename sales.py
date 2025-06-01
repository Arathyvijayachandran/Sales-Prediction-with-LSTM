import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import statsmodels.formula.api as smf

st.title("Sales Prediction with LSTM")

# File Upload
uploaded_file = st.file_uploader("Upload your sales data (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Data uploaded successfully!")
        st.write(df.head())

        # Data Preprocessing
        try:
            df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
        except ValueError as e:
            st.error(f"Error parsing date column: {e}")
            st.stop()

        df['date'] = df['date'].dt.year.astype('str') + '-' + df['date'].dt.month.astype('str') + '-01'
        df['date'] = pd.to_datetime(df['date'])
        df = df.groupby('date').sales.sum().reset_index()

        # Plot Sales
        st.subheader("Monthly Sales")
        fig = go.Figure([go.Scatter(x=df['date'], y=df['sales'])])
        st.plotly_chart(fig)

        # Differencing
        df_diff = df.copy()
        df_diff['prev_sales'] = df_diff['sales'].shift(1)
        df_diff = df_diff.dropna()
        df_diff['diff'] = df_diff['sales'] - df_diff['prev_sales']

        st.subheader("Monthly Sales Difference")
        fig = go.Figure([go.Scatter(x=df_diff['date'], y=df_diff['diff'])])
        st.plotly_chart(fig)

        # Add lag features
        df_supervised = df_diff.drop(['prev_sales'], axis=1)
        for inc in range(1, 13):
            df_supervised[f'lag_{inc}'] = df_supervised['diff'].shift(inc)
        df_supervised = df_supervised.dropna().reset_index(drop=True)

        # Regression
        model_reg = smf.ols('diff ~ ' + ' + '.join([f'lag_{i}' for i in range(1, 13)]), data=df_supervised).fit()
        st.write(f"Regression Adjusted R-squared: {model_reg.rsquared_adj:.4f}")

        # LSTM preparation
        df_model = df_supervised.drop(['sales', 'date'], axis=1)
        forecast_horizon = 6
        train_set, test_set = df_model[:-forecast_horizon].values, df_model[-forecast_horizon:].values

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(train_set)
        train_scaled = scaler.transform(train_set)
        test_scaled = scaler.transform(test_set)

        X_train, y_train = train_scaled[:, 1:], train_scaled[:, 0:1]
        X_test, y_test = test_scaled[:, 1:], test_scaled[:, 0:1]
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        # Build LSTM model (Fixed version)
        model_lstm = Sequential()
        model_lstm.add(LSTM(4, input_shape=(X_train.shape[1], X_train.shape[2])))
        model_lstm.add(Dense(1))
        model_lstm.compile(loss='mean_squared_error', optimizer='adam')
        model_lstm.fit(X_train, y_train, epochs=100, batch_size=1, verbose=0, shuffle=False)

        # Predict and invert transform
        y_pred = model_lstm.predict(X_test, batch_size=1)
        pred_combined = np.concatenate([y_pred, X_test.reshape(X_test.shape[0], X_test.shape[2])], axis=1)
        pred_inverted = scaler.inverse_transform(pred_combined)

        # Create forecasted results
        result_list = []
        sales_dates = list(df[-(forecast_horizon + 1):].date)
        act_sales = list(df[-(forecast_horizon + 1):].sales)

        for index in range(forecast_horizon):
            pred_value = int(pred_inverted[index][0] + act_sales[index])
            result_list.append({'date': sales_dates[index + 1], 'pred_value': pred_value})

        df_result = pd.DataFrame(result_list)
        df_sales_pred = pd.merge(df, df_result, on='date', how='left')

        # Plot results
        st.subheader("Sales Prediction")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_sales_pred['date'], y=df_sales_pred['sales'], name='Actual'))
        fig.add_trace(go.Scatter(x=df_sales_pred['date'], y=df_sales_pred['pred_value'], name='Predicted', line=dict(dash='dot')))
        st.plotly_chart(fig)

        # Show table
        st.subheader("Prediction Table")
        st.dataframe(df_sales_pred[['date', 'sales', 'pred_value']].dropna())

        # Download predictions
        csv = df_sales_pred.to_csv(index=False)
        st.download_button("Download Predictions", data=csv, file_name="sales_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"An error occurred: {e}")
