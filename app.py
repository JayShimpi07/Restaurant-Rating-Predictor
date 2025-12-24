import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Restaurant Rating Predictor",
    page_icon="🍽️",
    layout="wide"
)

# -------------------- Title --------------------
st.title("🍽️ Restaurant Rating Predictor")
st.markdown("Predict restaurant ratings using Machine Learning")

# -------------------- Upload Dataset --------------------
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")

    # -------------------- Show Dataset --------------------
    with st.expander("🔍 View Dataset"):
        st.dataframe(data.head())

    # -------------------- Preprocessing --------------------
    try:
        X = data[['Locality', 'Cuisines', 'Average Cost for two', 'Votes']]
        y = data['Aggregate rating']

        X = pd.get_dummies(X)

        scaler = StandardScaler()
        X[['Average Cost for two', 'Votes']] = scaler.fit_transform(
            X[['Average Cost for two', 'Votes']]
        )

        # -------------------- Train-Test Split --------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # -------------------- Model Training --------------------
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # -------------------- Evaluation --------------------
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        col1, col2 = st.columns(2)
        col1.metric("📉 RMSE", round(rmse, 4))
        col2.metric("📊 R² Score", round(r2, 2))

        # -------------------- Feature Importance --------------------
        importance = model.feature_importances_
        feature_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        st.subheader("🔥 Top 10 Influential Features")
        st.dataframe(feature_df.head(10))

        # -------------------- Plot --------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            feature_df.head(10)["Feature"],
            feature_df.head(10)["Importance"],
            color="crimson"
        )
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        ax.set_title("Top 10 Influential Features")

        st.pyplot(fig)

    except Exception as e:
        st.error("Something went wrong!")
        st.exception(e)

else:
    st.info("📌 Please upload a CSV file to start prediction.")
