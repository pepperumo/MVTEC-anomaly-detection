import streamlit as st
import pandas as pd
import torch

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import os
import pickle
import joblib
import cv2
from data_processing import (
    dataset_statistics, dataset_distribution_chart, load_dataset, 
    plot_bgr_pixel_densities, plot_pair_plots
)
from metrics_calculation import (
        load_evaluation_metrics,
        plot_roc_curve,
        plot_confusion_matrix
    )

from prediction import (
    run_inference_autoencoder, load_model_autoencoder, run_inference_knn, load_model_knn
)


# Overview Page
def overview_page():
    col1, col2, col3 = st.columns([1, 6, 1])
    
    st.title("🚀 Beyond Normal:  Unveiling Image Anomalies with AI")
    st.markdown("---")

    try:
        st.image("images/overview_image.png", use_container_width=True, 
                 caption="Normal vs. Anomalous Samples with Segmentation Masks")
    except FileNotFoundError:
        st.error("⚠️ Image file not found. Please check if 'images/overview_image.png' exists.")

    # Create three columns for key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div>
                <h2>5,450</h2>
                <p>High-Resolution Images</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div>
                <h2>15</h2>
                <p>Object Categories</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div>
                <h2>70+</h2>
                <p>Defect Types</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    
    st.header("🔍 What is Anomaly Detection?")
    st.write("""
        Imagine a world where machines can **spot defects** in products just like human inspectors—but **faster and with higher accuracy**!  
        This is exactly what **MVTec AD**, a powerful dataset, helps us achieve. It is designed for **automated quality control** in manufacturing by detecting **flaws** such as scratches, dents, and missing parts in different objects and textures.
        
        - 🖼️ **5,450 high-resolution images** across **15 object and texture categories**
        - ✅ **Training set**: Only contains **defect-free** images  
        - 🐞 **Test set**: Includes images with **over 70 different types of defects**  
        - 🎯 **Goal**: Automatically **detect and highlight anomalies** with **pixel-precise segmentation**
    """)

    st.subheader("🧐 **How Does Anomaly Detection Work?**")
    st.write("""
        The AI model learns what a **perfect product** looks like by studying thousands of **defect-free images**.  
        When it sees a **new image**, it checks:
        
        1️⃣ **Does this image match what I’ve seen before?**  
        2️⃣ **If not, where is the defect?**  

        The result? A **heatmap** showing the suspicious areas, along with a **segmentation mask** to pinpoint the defect.
    """)

    st.subheader("✨ **Bringing Anomalies to Light**: Real-World Examples")
    st.write("""
        Below are **three real examples** of AI-powered anomaly detection. Each image set follows this pattern:
        
        🔹 **First Column**: Original object with an anomaly  
        🔹 **Second Column**: AI-generated **heatmap** highlighting the defect  
        🔹 **Third Column**: Segmentation mask identifying **defect locations**  
        🔹 **Fourth Column**: **Ground truth** (expert-labeled defect areas for verification)  
    """)

    # Display anomaly detection images directly without additional data processing
    st.image("images/anomaly_visual_example_1.png", use_container_width=True, caption="Defective Wood - Liquid Stain")
    st.image("images/anomaly_visual_example_2.png", use_container_width=True, caption="Hazelnut - Hole Defect")
    st.image("images/anomaly_visual_example_3.png", use_container_width=True, caption="Leather - Cut Defect")

    st.write("""
        🔵 **The Heatmap (Second Column)**:  
        - AI scans the object and **highlights unusual areas** in **red/yellow**, indicating anomaly.  

        ⚫ **Segmentation Map (Third Column)**:  
        - Shows the **exact shape** of the detected anomaly, crucial for precise localization.

        ✅ **Ground Truth (Fourth Column)**:  
        - The manually labeled anomaly **used for AI validation**.

        This technology helps manufacturers **automate anomaly detection, reduce waste, and ensure top-tier product quality** at an industrial scale. 🚀  
    """)

    st.success("🌟 With AI-powered anomaly detection, we bring **precision and automation** to manufacturing quality control! 🔍✨")

    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Process Pipeline")
        st.markdown("""
            1. **Data Collection & Preprocessing**
            2. **Feature Extraction**
            3. **Model Training**
            4. **Anomaly Detection**
            5. **Results Visualization**
        """)
    
    with col2:
        st.subheader("🛠️ Key Technologies")
        st.markdown("""
            - **Deep Learning**: PyTorch
            - **Computer Vision**: OpenCV, PIL
            - **Data Analysis**: NumPy, Pandas
            - **Visualization**: Plotly, Matplotlib
        """)


# Dataset Page
def dataset_page():
    st.title("📊 Dataset Analysis & Exploratory Data")
    
    tab1, tab2 = st.tabs(["📈 Dataset Overview", "🎨 Feature Analysis"])
    
    with tab1:
        st.subheader("Dataset Structure")
        complete_df = load_dataset()
        if complete_df is not None:
            st.markdown("""
                <div>
                    <ul>
                        <li>Total Samples: {}</li>
                        <li>Categories: {}</li>
                    </ul>
                </div>
            """.format(
                len(complete_df),
                len(complete_df['category'].unique())
            ), unsafe_allow_html=True)
            
            with st.expander("🔍 View Full Dataset"):
                st.dataframe(complete_df, use_container_width=True)
        else:
            st.error("❌ Error: Dataset could not be loaded.")
    
        
        st.subheader("Statistical Analysis")
        df = dataset_statistics()
        if df is not None:
                        
            dataset_distribution_chart(df)
            
            with st.expander("📊 Detailed Statistics"):
                st.dataframe(df, use_container_width=True)
        else:
            st.error("❌ Error: Statistics could not be computed.")

    with tab2:
        st.subheader("Feature Engineering & Analysis")
        
        st.markdown("""
            <div>
                <h3>🔄 Dimensionality Reduction (PCA)</h3>
                <p>Our feature extraction pipeline includes:</p>
                <ul>
                    <li>Image preprocessing and normalization</li>
                    <li>Feature extraction using ResNet50</li>
                    <li>PCA transformation preserving 95% variance</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        complete_df = load_dataset()
        if complete_df is not None:
            selected_category = st.selectbox("Select a Category", 
                                     complete_df['category'].unique(), index=list(complete_df['category'].unique()).index('wood'))
            
            col1, col2 = st.columns(2)
            with col1:
                plot_bgr_pixel_densities(
                    complete_df[complete_df['category'] == selected_category],
                    pixel_columns=['num_pixels_b', 'num_pixels_g', 'num_pixels_r']
                )
            with col2:
                plot_pair_plots(complete_df[complete_df['category'] == selected_category])
        st.write("""
        **Key Takeaways from Feature Relationships:**
        
        **Strong Correlations:**  
        - The BGR pixel values exhibit **strong linear relationships**, which is expected as they represent color intensity.
        - **Perceived brightness** also shows a linear trend, confirming its dependence on RGB values.

        **Separation of Normal vs. Anomalous Data:**  
        - Some categories, like `Hazelnut` and `Tile`, show **clear separation** between normal (blue) and anomalous (red) points, indicating that anomalies have **distinct feature distributions**.
        - Other categories, such as `Screw` and `Transistor`, show **more overlap**, meaning their anomalies are harder to distinguish based only on pixel values.

        **Density Distributions:**  
        - Categories like `Carpet` and `Capsule` show **multi-modal distributions**, meaning that anomalies have **different types of defects**.
        - Categories like `Leather` and `Metal Nut` show anomalies with **different brightness levels**, suggesting that brightness-based anomaly detection could be effective.

        Overall, this analysis helps us understand which **features are useful for distinguishing anomalies** and which categories might need **additional feature engineering**.
        """)

        st.success("✅ Using PCA, BGR pixel distributions, and feature relationships, we ensure that the dataset is **optimized for training accurate anomaly detection models**.")

        

def synthetic_data_page():
    st.title("🔬 Data Enhancement Techniques")
    
    tab1, tab2 = st.tabs(["🧪 Synthetic Data", "🔄 Data Augmentation"])
    
    with tab1:
        st.header("Synthetic Data Generation")
        
        st.info("""
            🔍 **Synthetic Data** refers to artificially generated images that simulate anomalies 
            by modifying normal samples. Unlike data augmentation, synthetic data aims to create 
            new, realistic defect patterns that weren't present in the original dataset.
        """)
        st.write("""
        #### 🔍 Why Do We Need Synthetic Data?
        The **MVTec Anomaly Detection Dataset** contains **15 object categories**, but for each category:
        
        - The dataset is relatively **small**.
        - There are **far fewer anomaly images** than normal images.
        - Splitting test data for validation would leave even **less data** to train the model.

        To solve this problem, **we create additional "fake" anomaly images** to train the model better.  
        Instead of taking images of real defective objects (which are limited), we **manipulate normal images** by adding synthetic defects, such as **twisting, distorting, or overlaying textures**.
        
        """)
        st.image(
            "images/synthetic_example.png",
            caption="Synthetic Anomaly"
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div>
                    <h4>Synthetic Anomaly Generation</h4>
                    <ol>
                        <li><strong>Base Selection:</strong> Choose a normal image as base</li>
                        <li><strong>Defect Injection:</strong> Apply artificial defects:
                            <ul>
                                <li>Scratch patterns</li>
                                <li>Surface contamination</li>
                                <li>Structural deformations</li>
                                <li>Missing components</li>
                            </ul>
                        </li>
                        <li><strong>Validation:</strong> Ensure defect realism</li>
                        <li><strong>Integration:</strong> Add to validation dataset</li>
                    </ol>
                </div>
            """, unsafe_allow_html=True)
            
            st.success("""
                ✨ **Benefits of Synthetic Data**
                - Creates diverse anomaly patterns
                - Controls defect characteristics
                - Balances class distribution
                - Reduces data collection costs
            """)
        
        with col2:
                        
            st.warning("""
                ⚠️ **Important Considerations**
                - Synthetic defects must be realistic
                - Validation against real defects is crucial
                - Balance between synthetic and real data needed
            """)
    
    with tab2:
        st.header("Data Augmentation Techniques")
        
        st.info("""
            🔄 **Data Augmentation** applies label-preserving transformations to existing images
            to increase dataset variety and prevent overfitting. Unlike synthetic data, augmentation
            doesn't create new defect types but enhances model robustness through variations.
        """)

        st.write("""
        **Augmented data** is different from synthetic data. Instead of creating **new artificial images**, we **modify existing images** by applying **small transformations** like flipping, rotating, and resizing.  

        #### 🔍 Why Do We Need Data Augmentation?
        Even with synthetic data, **our dataset is still small** compared to what is needed for deep learning.  
        If we train a model on a **limited number of images**, the model might **memorize** the training data instead of **learning general patterns**. This is called **overfitting**.

        **To prevent overfitting, we increase the dataset size by applying transformations to images.**  
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div>
                <h5>🔲 Basic Transformations</h5>
                <ul>
                <li><strong>Resizing</strong>: Fixed 224×224 pixels</li>
                <li><strong>Horizontal Flip</strong>: 50% probability</li>
                <li><strong>Impact</strong>: Standardized input size</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div>
                <h5>🎨 Geometric Operations</h5>
                <ul>
                <li><strong>Rotation</strong>: 75% prob. (-90°, 90°, 180°)</li>
                <li><strong>Scaling & Translation</strong>: 75% probability</li>
                <li><strong>Impact</strong>: Position invariance</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div>
                <h5>🔄 Advanced Processing</h5>
                <ul>
                <li><strong>Gaussian Blur</strong>: Kernel size 3</li>
                <li><strong>Sigma Range</strong>: 0.01-0.05</li>
                <li><strong>Final Step</strong>: Tensor conversion</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📈 Augmentation examples")
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                "images/augmented_example.png",
                caption="Augmented Images Examples",
                use_container_width=True
            )
        with col2:
            st.image(
                "images/original_example.png",
                caption="Original Images Examples",
                use_container_width=True
            )
        
        st.success("""
            ✨ **Benefits of Data Augmentation**
            - Prevents overfitting
            - Improves model generalization
            - Increases effective dataset size
            - Maintains label validity
        """)

def resnet50_page():
    st.title("🔍 ResNet50 Feature Extraction")
    col1, col2= st.columns([3, 2])
    with col1:
        st.markdown("""
        ### Why Use a Pretrained Model (Transfer Learning)? 🧠
        Instead of starting from scratch, we take advantage of **ResNet50**, 
        a popular neural network that has already been trained on a large image dataset (ImageNet). 
        Because ResNet50 has “seen” many kinds of shapes, objects, and patterns,
        it has learned to recognize important features in images.

        By using these **pretrained features**, we:
        1. ⏱️ Save time and resources (no need to train a big model from zero).
        2. 🖼️ Gain access to a representation that already captures key visual patterns.
        3. 🎯 Focus on fine-tuning the model for our specific task (anomaly detection).
        """)

    with col2:
        st.image("images/resnet50.png", caption="Resnet50 Latent features extraction", use_container_width=True)

    st.markdown("""
        
        ### How We Extract Features from ResNet50
        We focus on two parts (or “blocks”) of ResNet50 and gather their outputs:
        - **Block 1**: Produces 512 features.
        - **Block 2**: Produces 1024 features.

        We then **combine** (concatenate) these for a total of **(512 + 1024) = 1536** features. 
        These numbers come from the internal layers of ResNet50.

        Essentially, these **1536 features** act like a summary of the image’s most important elements. 📝

        """)
def models_page():
    st.title("🤖 Anomaly Detection Models")

    tab1, tab2 = st.tabs(["KNN", "Autoencoder"])
    
    with tab1:
        st.header("KNN for Anomaly Detection") 
        col1, col2 = st.columns([2.5,2])
        with col1:
            st.markdown("""
        KNN (K-Nearest Neighbors) is a simple method that checks how "close" a new sample is 
        to existing samples. Here's the idea:
        1. We first collect "normal" images and extract their 1536 features. 
           We call this collection our **memory bank** of normal features. 🏦
        2. When a **test** image comes in:
           - ⚙️ We **extract** its 1536 features with the exact same process (ResNet blocks).
           - 📏 We measure its **distance** to each normal feature vector in the memory bank.
           - 🔎 We pick the **1 closest** neighbor (because k=1) and calculate the **average distance**.
             - If this average distance is **small**, it is likely "normal." ✅
             - If this average distance is **large**, it might be "anomalous" or unusual. 🐞
        """)
            st.image("images/k-nearest-neighbors-algorithm.png", caption="Visualizing the K-Nearest Neighbors approach")
        with col2:
            st.image("images/KNN_Pipeline.png", caption="KNN Anomaly Detection Pipeline")

  

    with tab2:
        st.header("Autoencoder for Anomaly Detection")
        col1, col2 = st.columns([2.5,2])
        with col1: 
            st.write("""
            Deep learning models have demonstrated remarkable performance in anomaly detection tasks, 
            particularly in complex scenarios where traditional methods often struggle. 
            These models can automatically learn intricate patterns and features from data, 
            making them highly effective at identifying subtle anomalies.
            """)

            st.write("""
            ### What is an autoencoder?

            An autoencoder is a specialized type of neural network designed to compress and reconstruct input images. 
            When applied to anomaly detection, the model is trained exclusively on normal images to learn the typical characteristics of the dataset.
            """)

            st.write("""
            ### Anomaly Detection Workflow

            #### 1. Feature Extraction
            - Collect a set of normal images.
            - Extract 1,536 features from each image using a **ResNet50** model.

            #### 2. Training Process
            - Train the autoencoder exclusively on normal data.
            - The autoencoder learns to efficiently encode and decode normal patterns.
            - The model optimizes its reconstruction error using normal samples.

            #### 3. Anomaly Detection
            - A new test image is passed through the trained autoencoder.
            - The reconstructed output is compared to the original input.
            - A high reconstruction error suggests an anomaly.
            - A low reconstruction error indicates that the sample is likely normal.
            """)

        st.success("""
        ## Key Advantages
        - **Fully Unsupervised Learning** – No need for labeled anomaly data.
        - **Ability to Capture Complex Normal Patterns** – The model generalizes well to unseen normal variations.
        - **Effective for High-Dimensional Image Data** – Works well with large and detailed datasets.
        """)
        st.warning("""
        ## Limitations
        - **Computationally Intensive Training** – Training deep autoencoders requires significant computational resources.
        - **Performance Sensitivity to Model Architecture** – The effectiveness depends heavily on model design.
        - **Difficulty Detecting Subtle Anomalies** – If anomalies resemble normal patterns closely, they may be overlooked.
        """)

        with col2:
            st.image("images/Autoencoder_Pipeline.png", caption="Autoencoder Anomaly Detection Pipeline", use_container_width=True)
            st.image("images/encoder_decoder.png", caption="Autoencoder structure", use_container_width=True)

def analysis_page():
    st.title("📊 Model Performance Analysis")
    
    # Move category selection outside of tabs
    selected_category = st.selectbox(
        "Select Category",
        ["bottle", "cable", "capsule", "carpet", "grid", 
            "hazelnut", "leather", "metal_nut", "pill", "screw",
            "tile", "toothbrush", "transistor", "wood", "zipper"],
        index=1  # Default to 'cable'
    )
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("KNN Model")
        try:
            confusion_matrices, roc_curves, auc_scores, f1_scores = load_evaluation_metrics('models/evaluation_metrics_knn.pkl')
            
            # Display ROC curve with unique key
            roc_fig = plot_roc_curve(selected_category, roc_curves, auc_scores)
            st.plotly_chart(roc_fig, use_container_width=True, key="knn_roc")

            # Display confusion matrix with unique key
            cm_fig = plot_confusion_matrix(selected_category, confusion_matrices, f1_scores)
            st.plotly_chart(cm_fig, use_container_width=True, key="knn_cm")
            
        except FileNotFoundError:
            st.error("KNN evaluation metrics file not found. Please run model evaluation first.") 
        except Exception as e:
            st.error(f"Error loading KNN metrics: {str(e)}")

    with col2:
        st.subheader("Autoencoder Model")
        try:
            confusion_matrices, roc_curves, auc_scores, f1_scores = load_evaluation_metrics('models/evaluation_metrics_autoencoder.pkl')
            
            # Display ROC curve with unique key
            roc_fig = plot_roc_curve(selected_category, roc_curves, auc_scores)
            st.plotly_chart(roc_fig, use_container_width=True, key="ae_roc")

            # Display confusion matrix with unique key
            cm_fig = plot_confusion_matrix(selected_category, confusion_matrices, f1_scores)
            st.plotly_chart(cm_fig, use_container_width=True, key="ae_cm")
            
        except FileNotFoundError:
            st.error("Autoencoder evaluation metrics file not found. Please run model evaluation first.")
        except Exception as e:
            st.error(f"Error loading Autoencoder metrics: {e}")



def prediction_page():
    """
    Streamlit page to view anomaly detection images.
    """
    st.title("🔍 Anomaly Detection - Image Viewer")

    st.info("Select a category and image to view the anomaly detection result.")

    # Select a category
    selected_category = st.selectbox(
        "Select a Category",
        ["bottle", "cable", "capsule", "carpet", "grid", 
         "hazelnut", "leather", "metal_nut", "pill", "screw",
         "tile", "toothbrush", "transistor", "wood", "zipper"],
        key="shared_category"  # Unique key for each selectbox
    )

    # List available images in the selected category
    category_dir_knn = os.path.join("images/Dataset_knn", selected_category)
    category_dir_autoencoder = os.path.join("images/Dataset_autoencoder", selected_category)

    available_images_knn = []
    if os.path.exists(category_dir_knn):
        available_images_knn = [f for f in os.listdir(category_dir_knn) if os.path.isfile(os.path.join(category_dir_knn, f))]
    
    available_images_autoencoder = []
    if os.path.exists(category_dir_autoencoder):
        available_images_autoencoder = [f for f in os.listdir(category_dir_autoencoder) if os.path.isfile(os.path.join(category_dir_autoencoder, f))]

    # Find common images
    available_images = list(set(available_images_knn) & set(available_images_autoencoder))

    selected_image = st.selectbox(
        "Select an Image", 
        available_images,
        key="shared_image"  # Unique key for each selectbox
    )

    run_prediction = st.button("Run Prediction")

    tab1, tab2 = st.tabs(["KNN", "Autoencoder"])

    with tab1:
        st.subheader("KNN Model")
        if run_prediction:
            display_image("knn", selected_category, selected_image)

    with tab2:
        st.subheader("Autoencoder Model")
        if run_prediction:
            display_image("autoencoder", selected_category, selected_image)

def display_image(model_type, selected_category, selected_image):
    """
    Helper function to display a single image based on the selected model type.
    """
    if selected_image:
        # Construct file paths
        image_path = os.path.join(f"images/Dataset_{model_type}", selected_category, selected_image)

        # Display image
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                st.image(image, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying {model_type} image: {e}")
        else:
            st.error(f"{model_type} Image not found: {image_path}")
    
    st.markdown("""
    ### Explanation of the displayed image:
    
    1️⃣ **Top Left: Original Image**
    - This is the raw image from the dataset.
    - The object is analyzed to detect potential anomalies.
    
    2️⃣ **Top Right: Heatmap**
    - The heatmap represents the anomaly score distribution.
    - Color Legend:
        - 🔴 Red/Yellow: High anomaly score (defective region).
        - 🔵 Blue: Normal areas with low anomaly probability.
    - The defect region is highlighted based on the anomaly model’s prediction.
    
    3️⃣ **Bottom Left: Segmentation Map**
    - This is a binary mask that highlights the detected defect areas.
    - White pixels represent anomalous regions identified by the model.
    - It is created by thresholding the heatmap to localize defects.
    
    4️⃣ **Bottom Right: Ground Truth**
    - The ground truth mask is a manually labeled reference.
    - It defines the true defective areas for validation.
    - The segmentation map should ideally match this mask for accurate detection.
    """)



def conclusion_and_improvements_page():
    st.title("🏁 Conclusion and Future Improvements")

    st.write("""
    ## Conclusion

    Throughout the project, several Machine Learning and Deep Learning models have been applied to the available image data. We compared different models and approaches in data preparation to optimize anomaly detection performance.

    The evaluation shows that the Convolutional Autoencoder and the KNN approach significantly outperform other models, and both yield better results using the deep feature extraction approach.

    ### Key Takeaways

    - **Deep Feature Extraction**: Significantly outperforms manual extracted features, as seen in the consistently better performance of ResNet50-based methods.
    - **ResNet50 Block Performance**: Two evaluated deep feature extraction approaches, one using ResNet50 blocks 1 and 2, the other using block 3, both seem similarly suitable on average over all categories. However, within single categories, one often showed significantly better performance than the other.
    - **KNN Approach**: Leverages memory-based similarity comparisons, providing a robust alternative to parametric models.
    - **Autoencoder Approach**: Deep learning using an autoencoder provides strong reconstruction-based anomaly detection, ensuring comprehensive feature learning.
    - **Synthetic Validation Data**: Introduced diversity and improved cross-validation but also posed challenges, as some anomalies may not perfectly mimic real-world defects.

    ## Future Improvements

    - **Data Augmentation Refinement**: Exploring more advanced augmentation techniques, such as generative models (e.g., GANs), could enhance training diversity.
    - **Model Ensembling**: Combining multiple anomaly detection models could improve robustness and generalization.
    - **Hyperparameter Optimization**: Fine-tuning model hyperparameters further, using techniques like Bayesian optimization, could boost performance.
    - **Alternative Architectures**: Exploring transformer-based architectures or vision encoders like ViTs for anomaly detection could yield even better results.
    - **Better Synthetic Data**: Refining the process of synthetic anomaly generation to better align with real-world defects.

    Overall, the introduced approaches provide a solid foundation for industrial anomaly detection tasks. Further optimizations and explorations in feature extraction and model selection could push performance even higher in future studies.
    """)

# Bibliography Page
def bibliography_page():
    st.title("📚 Bibliography & References")
    
    st.markdown("""
        ### Core Papers & Methods
        
        1. Liu, J., Xie, G., Wang, J., Li, S., Wang, C., Zheng, F., & Jin, Y. (2023). 
        _Deep Industrial Image Anomaly Detection: A Survey_. Springer Nature.  
        [arXiv:2301.11514](https://arxiv.org/abs/2301.11514)

        2. Yang, J., Shi, Y., & Qi, Z. (2020).
        _DFR: Deep Feature Reconstruction for Unsupervised Anomaly Segmentation_.  
        [arXiv:2012.07122](https://arxiv.org/abs/2012.07122)

        3. Bühler, J., Fehrenbach, J., Steinmann, L., Nauck, C., & Koulakis, M. (2024).
        _Domain-independent detection of known anomalies_. Karlsruhe Institute of Technology (KIT) & preML GmbH.  
        [arXiv:2407.02910](https://arxiv.org/abs/2407.02910)

        4. Heckler, L., & König, R. (2024).
        _Feature Selection for Unsupervised Anomaly Detection and Localization Using Synthetic Defects_.
        MVTec Software GmbH & Technical University of Munich.
        In Proceedings of the 19th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2024), 154-165.  
        [DOI: 10.5220/0012385500003660](https://doi.org/10.5220/0012385500003660)

        5. Rippel, O., Mertens, P., & Merhof, D. (2020).
        _Modeling the Distribution of Normal Data in Pre-Trained Deep Features for Anomaly Detection_.
        RWTH Aachen University.  
        [arXiv:2005.14140](https://arxiv.org/abs/2005.14140)

        6. Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019).
        _MVTec AD – A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection_.
        MVTec Software GmbH.  
        [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

        7. Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P. (2022).
        _Towards Total Recall in Industrial Anomaly Detection_.
        University of Tübingen & Amazon AWS.  
        [arXiv:2106.08265](https://arxiv.org/abs/2106.08265)

        8. Zheng, Y., Wang, X., Qi, Y., Li, W., & Wu, L. (2022).
        _Benchmarking Unsupervised Anomaly Detection and Localization_.
        University of Chinese Academy of Sciences, SenseTime Research, & Tsinghua University.  
        [arXiv:2205.14852](https://arxiv.org/abs/2205.14852)
    """)

def main():
    with st.sidebar:
        st.image("images/logo.png", width=100)  # Add your logo here
        st.title("Navigation")
        
        
        selection = st.radio(
            "Select a Section",
            ["Overview", 
             "Dataset & EDM",
             "Synthetic Data & Augmentation",
             "Transfer Learning - Resnet50",
             "Models",
             "Analysis",
             "Prediction",
             "Conclusion and Improvements",
             "Bibliography"],
            format_func=lambda x: f" {x}"
        )
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; color: #666;'>
            
            <div style='margin: 10px 0;'>
            This app is maintained by:<br>
            <a href="https://www.linkedin.com/in/giuseppe-rumore-b2599961" target="_blank">Giuseppe Rumore</a> |
            <a href="https://www.linkedin.com/in/micaela-w%C3%BCnsche-9baaa710b/" target="_blank">Micaela Wünsche</a> |
            <a href="https://www.linkedin.com/in/majid-jafari-62909071/" target="_blank">Majid Jafari</a>
            </div>
            </a>
            <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Logo.svg.original.svg" 
             width="80" 
             alt="LinkedIn"
             style="margin-top: 10px;">
            </a>
            <br>
            <a href="https://github.com/DataScientest-Studio/nov24_bds_mvtec-anomaly-detection" target="_blank">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
             width="40"
             alt="GitHub"
             style="margin-top: 10px; border-radius: 50%;">
            </a>
            <div style='text-align: center; color: #666;'>
            <small>Version 1.0.0</small><br>
            <small>© 2025 MVTec Anomaly Detection</small><br>
            </div>
        """, unsafe_allow_html=True)
        
    if selection == "Overview":
        overview_page()
    elif selection == "Dataset & EDM":
        dataset_page()
    elif selection == "Synthetic Data & Augmentation":
        synthetic_data_page()
    elif selection == "Transfer Learning - Resnet50":
        resnet50_page()
    elif selection == "Models":
        models_page()
    elif selection == "Analysis":
        analysis_page()
    elif selection == "Prediction":
        prediction_page()
    elif selection == "Conclusion and Improvements":
        conclusion_and_improvements_page()
    elif selection == "Bibliography":
        bibliography_page()

if __name__ == "__main__":
    main()
