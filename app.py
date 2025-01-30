import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(layout="wide")

def load_data():
    df = pd.read_csv("Otodom_Flat_Listings.csv")
    df['Price_per_m2'] = (df['Price'] / df['Surface']).round(2)
    df_cleaned = df.drop(columns=['Title', 'Location', 'Link', 'Finishing_Condition', 'Balcony_Garden_Terrace', 'Heating', 'City', 'Floor'])
    df_cleaned = df_cleaned.dropna(subset=['Voivodeship'])
    df_cleaned = df_cleaned[(df_cleaned['Price'] <= 3_900_000) & (df_cleaned['Price'] >= 100_000) & (df_cleaned['Surface'] <= 200) & (df_cleaned['Surface'] >= 10)]
    df_cleaned = df_cleaned[df_cleaned['Price_per_m2'] > 0]
    df_cleaned = df_cleaned[df_cleaned['Number_of_Rooms'] != 'więcej niż 10']
    df_cleaned['Number_of_Rooms'] = df_cleaned['Number_of_Rooms'].astype(int)
    df_cleaned['Parking_Space'] = df_cleaned['Parking_Space'].notna().astype(int)
    return df, df_cleaned

# Wczytanie danych
df_raw, df = load_data()

# Strona główna
st.title("Dashboard Analizy Cen Mieszkań - Otodom")

# Wyświetlenie tabeli
tab1, tab2, tab3, tab4 = st.tabs(["Dane", "Wizualizacje", "Modelowanie", "Mapa Województw"])

with tab1:
    st.header("Podgląd danych")
    data_option = st.radio("Wybierz dane do wyświetlenia:", ["Dane przed czyszczeniem", "Dane po czyszczeniu"])
    
    if data_option == "Dane przed czyszczeniem":
        st.write(df_raw.head())
    else:
        st.write(df.head())
    
    if st.checkbox("Pokaż statystyki podstawowe"):
        stats_option = st.radio("Wybierz dane do analizy statystycznej:", ["Dane przed czyszczeniem", "Dane po czyszczeniu"])
        if stats_option == "Dane przed czyszczeniem":
            st.write(df_raw.describe(include=[np.number]))
        else:
            st.write(df.describe(include=[np.number]))

with tab2:
    st.header("Wizualizacje")
    col1, col2 = st.columns(2)
    
    # with col1:
    fig = px.scatter(df, x='Surface', y='Price', color='Voivodeship', title='Cena vs Powierzchnia', hover_data=['Price_per_m2'], width=1200)
    st.plotly_chart(fig)
    
    # with col2:
    voivodeship_counts = df['Voivodeship'].value_counts().reset_index()
    voivodeship_counts.columns = ['Voivodeship', 'Count']
    fig = px.bar(voivodeship_counts, x='Voivodeship', y='Count', title='Liczba ogłoszeń na województwo')
    st.plotly_chart(fig)
    
    fig = px.histogram(df, x="Price", title="Histogram cen")
    st.plotly_chart(fig)
    
    fig = px.histogram(df, y="Number_of_Rooms", color="Number_of_Rooms", labels={'Number_of_Rooms': 'Number of rooms'}, title="Rozkład liczby pokoi")
    st.plotly_chart(fig)
    
    fig = px.histogram(df, x="Parking_Space", color="Parking_Space", labels={'Parking_Space': 'Parking Space'}, title="Rozkład dostępności parkingu")
    st.plotly_chart(fig)
    
    mean_price_per_m2 = df.groupby('Voivodeship')['Price_per_m2'].mean().reset_index()
    mean_price_per_m2 = mean_price_per_m2.sort_values(by='Price_per_m2', ascending=False)
    fig = px.bar(mean_price_per_m2, x='Voivodeship', y='Price_per_m2', title='Średnia cena za m² w województwach', labels={'Voivodeship': 'Voivodeship', 'Price_per_m2': 'Average Price per m²'}, color='Price_per_m2', color_continuous_scale='Viridis')
    st.plotly_chart(fig)
    
    fig = px.scatter(df, x='Surface', y='Price_per_m2', color='Price_per_m2', title='Powierzchnia vs Cena za m²', labels={'Surface': 'Surface (m²)', 'Price_per_m2': 'Price per m²'}, size='Price_per_m2', hover_data=['Number_of_Rooms'],color_continuous_scale='Viridis')
    st.plotly_chart(fig)
    
    fig = px.scatter(df, x='Number_of_Rooms', y='Price', color='Price_per_m2', title='Cena vs Liczba pokoi', labels={'Number_of_Rooms': 'Number of Rooms', 'Price': 'Price (PLN)'}, hover_data=['Surface'],color_continuous_scale='Viridis')
    st.plotly_chart(fig)
    
with tab3:
    st.header("Modelowanie")
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_features = encoder.fit_transform(df[['Voivodeship', 'Parking_Space', 'Number_of_Rooms']])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Voivodeship', 'Parking_Space', 'Number_of_Rooms']))
    df_encoded = pd.concat([df.drop(columns=['Voivodeship', 'Parking_Space', 'Number_of_Rooms']).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    X = df_encoded.drop(columns=['Price_per_m2'])
    y = df_encoded['Price_per_m2']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_choice = st.selectbox("Wybierz model", ["Random Forest", "Regresja Liniowa"])
    
    if model_choice == "Random Forest":
        model = RandomForestRegressor(random_state=42, n_estimators=100)
    else:
        model = LinearRegression()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    cross_val_mse = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5).mean()
    cross_val_rmse = np.sqrt(cross_val_mse)
    cross_val_r2 = cross_val_score(model, X, y, scoring='r2', cv=5).mean()
    

    fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Rzeczywiste ceny za m²", "y": "Przewidywane ceny za m²"}, title="Rzeczywiste vs. Przewidywane ceny")
    fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color="red"))
    st.plotly_chart(fig)
    
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**R²:** {r2:.2f}")
    st.write(f"**RMSE:** {np.sqrt(mse)}")
    st.write(f"**Cross-Validation MSE:** {cross_val_mse:.2f}")
    st.write(f"**Cross-Validation R²:** {cross_val_r2:.2f}")
    st.write(f"**Cross-Validation RMSE:** {cross_val_rmse:.2f}")
    
with tab4:
    st.header("Mapa Województw")
    shapefile_path = 'wojewodztwa.shp'
    voivodeship_gdf = gpd.read_file(shapefile_path)
    voivodeship_gdf['JPT_NAZWA_'] = voivodeship_gdf['JPT_NAZWA_'].str.lower()
    average_price_per_sqm = df.groupby('Voivodeship')['Price_per_m2'].mean().reset_index()
    average_price_per_sqm['Voivodeship'] = average_price_per_sqm['Voivodeship'].str.lower()
    average_price_per_sqm.columns = ['JPT_NAZWA_', 'avg_price_per_sqm']
    voivodeship_gdf = voivodeship_gdf.merge(average_price_per_sqm, on='JPT_NAZWA_', how='left')
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    voivodeship_gdf.plot(column='avg_price_per_sqm', cmap='PuBuGn', legend=True, legend_kwds={'label': "Average Price per sqm (PLN)"}, ax=ax)
    for x, y, label, price in zip(voivodeship_gdf.geometry.centroid.x, voivodeship_gdf.geometry.centroid.y, voivodeship_gdf['JPT_NAZWA_'], voivodeship_gdf['avg_price_per_sqm']):
        ax.text(x, y, f"{label.capitalize()}\n{price:.2f} PLN/m²", fontsize=10, ha='center', color='black')
    st.pyplot(fig)

# st.sidebar.header("Opcje filtrowania")
# st.sidebar.selectbox("Wybierz województwo", df['Voivodeship'].unique())
