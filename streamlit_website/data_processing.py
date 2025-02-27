import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from joypy import joyplot
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load dataset
def load_dataset():
    file_path = "Data/mvtec_meta_features_dataset.csv"
    try:
        complete_df = pd.read_csv(file_path)

        # Show available column names for debugging
        print("Available columns:", complete_df.columns)

        # Verify column presence
        required_columns = ["category", "set_type", "anomaly_status"]
        for col in required_columns:
            if col not in complete_df.columns:
                raise KeyError(f"Missing required column: {col}")
        
        # Define the subclasses for each category
        subclasses = {
            'Texture-Based': ['carpet', 'wood', 'tile', 'leather', 'zipper'],
            'Industrial Components': ['cable', 'transistor', 'screw', 'grid', 'metal_nut'],
            'Consumer Products': ['bottle', 'capsule', 'toothbrush'],
            'Edible': ['hazelnut', 'pill']
        }

        # Add a new column to the DataFrame to store the subclass
        complete_df['subclass'] = complete_df['category'].apply(
            lambda x: next((key for key, value in subclasses.items() if x in value), 'Unknown')
        )

        # Reorder columns to place 'subclass' after 'category'
        cols = list(complete_df.columns)
        cols.insert(cols.index('category') + 1, cols.pop(cols.index('subclass')))
        complete_df = complete_df[cols]

        return complete_df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Function to generate dataset statistics
def dataset_statistics():
    df = load_dataset()
    if df is not None:
        print("Loaded dataset preview:\n", df.head())  # Debugging step

        # Aggregate counts for each category and condition
        train_normal = df[(df['set_type'] == 'train') & (df['anomaly_status'] == 'normal')].groupby('category').size()
        test_normal = df[(df['set_type'] == 'test') & (df['anomaly_status'] == 'normal')].groupby('category').size()
        test_anomalous = df[(df['set_type'] == 'test') & (df['anomaly_status'] == 'anomalous')].groupby('category').size()

        # Combine into a single DataFrame
        final_summary = pd.DataFrame({
            'Train Normal Images': train_normal,
            'Test Normal Images': test_normal,
            'Test Anomalous Images': test_anomalous
        }).fillna(0).reset_index()

        return final_summary
    return None

# Function to generate the bar chart
def dataset_distribution_chart(df):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['category'], 
        y=df['Train Normal Images'], 
        name='Train Normal Images',
        marker_color='blue'
    ))
    fig.add_trace(go.Bar(
        x=df['category'], 
        y=df['Test Normal Images'], 
        name='Test Normal Images',
        marker_color='red'
    ))
    fig.add_trace(go.Bar(
        x=df['category'], 
        y=df['Test Anomalous Images'], 
        name='Test Anomalous Images',
        marker_color='green'
    ))

    # Update layout
    fig.update_layout(
        title="Distribution of Normal and Anomalous Images per Category",
        xaxis_title="Categories",
        yaxis_title="Number of Images",
        barmode='stack',
        legend_title="Image Types"
    )

    # Display chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Function to display the complete dataframe with expander
def display_dataframe():
    df = load_dataset()
    if df is not None:
        with st.expander("Show Complete DataFrame"):
            st.dataframe(df)



def plot_bgr_pixel_densities(df, pixel_columns=['num_pixels_b', 'num_pixels_g', 'num_pixels_r']):
    """
    Generate JoyPy density plots for pixel counts of BGR channels for a given category.

    Parameters:
        df (pd.DataFrame): Filtered DataFrame for a single category.
        pixel_columns (list): List of column names for BGR pixel counts.

    Returns:
        None
    """
    if df.empty:
        st.warning("⚠️ No data available for the selected category.")
        return

    # Plot JoyPy density plot
    fig, axes = joyplot(
        data=df,
        by="category",  # Group by category
        column=pixel_columns,
        color=['blue', 'green', 'red'],  # Colors for BGR channels
        alpha=0.5,
        fade=True,
        legend=True,
        linewidth=1.0,
        overlap=3,
        figsize=(8, 6)  # Adjust the figure size here
    )

    # Add title and labels
    plt.title(f'Density Plots for {df["category"].unique()[0]}', fontsize=14)
    plt.xlabel('Number of Pixels Density', fontsize=12)
    plt.ylabel('Categories', fontsize=12)

    # Show the plot in Streamlit
    st.pyplot(fig)

    
def plot_pair_plots(complete_df):
    """
    Generate and display pair plots for each category in the dataset.

    Parameters:
        complete_df (pd.DataFrame): The input DataFrame containing image features and categories.

    Returns:
        None
    """

    # Define the features to be included in the pairplot
    features = ['num_pixels_b', 'num_pixels_g', 'num_pixels_r', 'perceived_brightness']

    # Create a separate pairplot for each category
    for category in complete_df['category'].unique():
        # Filter data for current category
        category_df = complete_df[complete_df['category'] == category]
        
        # Check if the filtered DataFrame is not empty
        if not category_df.empty:
            # Create PairGrid with hue and palette
            g = sns.PairGrid(category_df, vars=features, hue='anomaly_status', palette={'normal': 'blue', 'anomalous': 'red'})
            
            # Map the plots to the grid
            g.map_upper(sns.scatterplot, alpha=0.6)
            g.map_diag(sns.histplot, kde=True)
            g.map_lower(sns.scatterplot, alpha=0.6)  
            
            # Add legend
            g.add_legend()
            
            # Customize the plot
            g.figure.suptitle(f'Feature Relationships for {category.title()}', y=1.02, fontsize=14)
            
            # Improve label readability
            for i in range(len(g.axes)):
                for j in range(len(g.axes)):
                    if g.axes[i][j] is not None:
                        g.axes[i][j].set_xlabel(g.axes[i][j].get_xlabel().replace('_', ' ').title())
                        g.axes[i][j].set_ylabel(g.axes[i][j].get_ylabel().replace('_', ' ').title())
            
            # Adjust legend position to the right without overlapping the plots
            g._legend.set_bbox_to_anchor((1.05, 0.5))
            g._legend.set_loc('center left')
            
            plt.tight_layout()
            st.pyplot(g.figure)

