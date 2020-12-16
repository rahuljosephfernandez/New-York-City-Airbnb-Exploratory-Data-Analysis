from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly_express as px
import plotly.io as pio 
import streamlit as st
import pandas as pd 
import numpy as np

# HEADER

st.title("New York City Airbnb Exploration")
st.write("## Explore NYC's Airbnb Listings")

# CSS

def css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css("style.css")

# SIDEBAR CONTENT

#

# DATA IMPORTS

data_path = (
    "listings.csv"
)

@st.cache(persist=True)
def load_listings():

    listings = pd.read_csv(data_path)

    airbnb = listings.drop(['id','listing_url','scrape_id','last_scraped','name','description','neighborhood_overview','picture_url',
    'host_id','host_url','host_name','host_since','host_location','host_about','host_response_rate','host_response_time','host_acceptance_rate',
    'host_thumbnail_url','host_picture_url','host_neighbourhood','host_verifications','host_has_profile_pic','neighbourhood','neighbourhood_cleansed',
    'property_type','bathrooms','bathrooms_text','amenities','calendar_updated','calendar_last_scraped','license','first_review','last_review'],
     axis = 1)

    prc0 = airbnb['price'].str.replace("$","")
    prc1 = prc0.str.replace(",","")
    prcs = prc1.astype(float)
    
    airbnb['price'] = prcs

    return(airbnb)

dataset = load_listings()

# BOROUGH PLOT

st.subheader("")
st.title("Exploratory Data Analysis")
st.title("")
st.write("### Select Borough:")
boroughs = pd.Series(dataset['neighbourhood_group_cleansed'].unique())
bhs = st.selectbox("",options = boroughs)

filtered_data = pd.DataFrame(dataset[dataset['neighbourhood_group_cleansed'] == bhs])
filtered_data['latitude'] = pd.to_numeric(filtered_data['latitude'],errors = 'coerce')
filtered_data['longitude'] = pd.to_numeric(filtered_data['longitude'],errors = 'coerce')

coords = {'lat': filtered_data['latitude'], 'lon': filtered_data['longitude']}
map_coords = pd.DataFrame(data=coords)

st.write("##### Median Price for", bhs, "-", np.median(filtered_data['price']))
st.write("")
st.map(map_coords)


st.write("##### Show Raw Data:")
if st.checkbox("", False):
    st.write(filtered_data)

st.markdown("")

# CORRESPONDING PRICES HISTOGRAM

st.write("### Price Distribution for", bhs)
hs = px.histogram(filtered_data,
                 filtered_data['price'],
                 labels = {"price": "Price ($)",
                            "y": "Count"},
                 template = 'plotly_dark',
                 width = 750)
st.write(hs)

# BOROUGH BOX PLOTS

st.write("### Borough Price Distributions")
vp = px.box(filtered_data,
               y = dataset['price'],
               color = dataset['neighbourhood_group_cleansed'],
               template = 'plotly_dark',
               range_y = (0, 500),
               labels = {"color": "Borough",
                         "y": "Price ($)"},
               width = 750)
st.write(vp)

# FEATURE PLOTTING

st.title("")
st.subheader("")
st.title("Dimensionality Reduction")

# Smaller Feature Set
features = ['host_listings_count','bedrooms','beds','price',
            'minimum_nights','maximum_nights','number_of_reviews',
            'calculated_host_listings_count']

# Main Feature Set
feature_set = dataset.select_dtypes(include = ['int64','float64'])
for col in feature_set.columns:
    feature_set[col] = feature_set[col].fillna((feature_set[col].mean()))

st.title("")
st.write("### Feature Scatter-Plot Matrix")

labs = {
    str(i): ''
    for i in feature_set.columns
}
fnum = st.slider("", min_value = 1, max_value = 35, value = 7)
sm = px.scatter_matrix(dataset,
                       dimensions = feature_set.iloc[:, 0:fnum],
                       color = dataset['neighbourhood_group_cleansed'],
                       labels = labs,
                       template = 'plotly_dark',
                       width = 750,
                       height = 600)
sm.update_xaxes(title_font=dict(size=1))
sm.update_yaxes(title_font=dict(size=1))

st.write(sm)

# PRINCIPAL COMPONENT ANALYSIS (PCA)

    # PCA ON FEATURE SET

st.subheader("")
st.write("### Principal Components Scatter-Plot Matrix")
pca = PCA()
components = pca.fit_transform(feature_set)
flabels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

fnums = st.slider("", min_value = 1, max_value = 35, value = 4)
pc = px.scatter_matrix(components,
                        labels = flabels,
                        dimensions = range(fnums),
                        color = dataset['neighbourhood_group_cleansed'],
                        template = 'plotly_dark'
                        )
st.write(pc)
if st.checkbox("Show Principal Components", False):
    st.write(components)

# PRINCIPAL COMPONENTS EXPLAINED VARIANCE

pcafit = pca.fit(feature_set)
expvar = pca.explained_variance_ratio_

st.subheader("")
st.write("### Principal Components Explained Variance")
exp = px.area(
    x = range(0, 35),
    y = expvar,
    template = 'plotly_dark',
    height = 700,
    labels = {"x": "Principal Components", "y": "Explained Variance"}
        )
st.write(exp)

# 3-DIMENSIONAL PRINCIPAL COMPONENTS EXPLORATION

pca3 = PCA(n_components=3)
components3 = pca3.fit_transform(feature_set)

totalvar = pca3.explained_variance_ratio_.sum() * 100

st.title("")
st.subheader("Visualizing the Principal Components in 3 Dimensions")
pc3 = px.scatter_3d(
    components3,
    x = components3[:,0],
    y = components3[:,1],
    z = components3[:,2],
    color = feature_set['price'],
    size = feature_set['price'],
    title=f'Total Explained Variance: {totalvar:.2f}%',
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'},
    template = 'plotly_dark'
)
st.write(pc3)
if st.checkbox("Show Principal Components (N = 3)", False):
    st.write(components3)

# 2-DIMENSIONAL PRINCIPAL COMPONENTS EXPLORATION

pca2 = PCA(n_components = 2)
components2 = pca2.fit_transform(feature_set)

totalvar2 = pca2.explained_variance_ratio_.sum() * 100

st.title("")
st.subheader("Visualizing the Principal Components in 2 Dimensions")
pc2 = px.scatter(
    components3,
    x = components3[:,0],
    y = components3[:,1],
    title=f'Total Explained Variance: {totalvar2:.2f}%',
    labels={'x': 'PC 1', 'y': 'PC 2'},
    template = 'plotly_dark'
)
st.write(pc2)
if st.checkbox("Show Principal Components (N = 2)", False):
    st.write(components2)

#