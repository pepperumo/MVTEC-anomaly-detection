import pandas as pd
import numpy as np
import cv2
import os
import pickle
from skimage.measure import shannon_entropy
from skimage.feature import blob_log
from skimage.restoration import estimate_sigma
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.transform import hough_line, hough_line_peaks
from scipy.fftpack import fft2, fftshift
from tqdm import tqdm
import torch

# get list of all categories
def get_category_list(path_to_mvtec_dataset_dir):
    # return list of all subfolder names
    categories = [category for category in os.listdir(path_to_mvtec_dataset_dir) if os.path.isdir(os.path.join(path_to_mvtec_dataset_dir, category))]
    # remove subfolder name 'pickle_files', if present
    if 'pickle_files' in categories:
        categories.remove('pickle_files')

    return categories

# Get all image and ground_truth paths for all categories
def get_all_image_and_ground_truth_paths(path_to_mvtec_dataset_dir, category):
    train_path = os.path.join(path_to_mvtec_dataset_dir, category, 'train')
    test_path = os.path.join(path_to_mvtec_dataset_dir, category, 'test')
    gt_path = os.path.join(path_to_mvtec_dataset_dir, category, 'ground_truth')
    
    def get_image_paths(root_path):
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        image_paths = []
        for dirpath, _, filenames in os.walk(root_path):
            for filename in filenames:
                if filename.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(dirpath, filename))
        return image_paths
    
    # TODO: check difference to previous function?
    def get_ground_truth_paths(root_path):
        ground_truth_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        ground_truth_paths = []
        for dirpath, _, filenames in os.walk(root_path):
            for filename in filenames:
                if filename.lower().endswith(ground_truth_extensions):
                    ground_truth_paths.append(os.path.join(dirpath, filename))
        return ground_truth_paths
    
    # Get image and ground_truth paths
    train_image_paths = get_image_paths(train_path)
    test_image_paths = get_image_paths(test_path)
    ground_truth_paths = get_ground_truth_paths(gt_path)
    
    return train_image_paths, test_image_paths, ground_truth_paths, train_image_paths+test_image_paths

# resize images to 224x224, returns list of images
def resize_images(image_paths, target_size=(224, 224)):
    resized_images = {}
    for category, paths in image_paths.items():
        print('    ', category)
        resized_images[category] = []
        for path in paths:
            image = cv2.imread(path)
            if image is not None:
                resized_image = cv2.resize(image, target_size)
                resized_images[category].append(resized_image)
    return resized_images

# return list of dictionaries containing metadata for each images adressed in image_paths
def extract_image_metadata(image_paths):
    metadata = []
    for category, paths in image_paths.items():
        print('    ', category)
        for path in paths:
            if 'train' in path:
                set_type = 'train'
            elif 'test' in path:
                set_type = 'test'
            elif 'validation' in path:
                set_type = 'validation'
            else:
                set_type = 'unknown'

            if 'good' in path:
                anomaly_status = 'normal'
                anomaly_type = 'none'
            else:
                anomaly_status = 'anomalous'
                anomaly_type = os.path.basename(os.path.dirname(path))

            image = cv2.imread(path)
            if image is not None:
                height, width, _ = image.shape
                aspect_ratio = width / height

                metadata.append({
                    'category': category,
                    'set_type': set_type,
                    'anomaly_status': anomaly_status,
                    'anomaly_type': anomaly_type,
                    'width': width,
                    'height': height,
                    'aspect_ratio': aspect_ratio
                })

    return metadata

# return dictionary with various features computed from given image
def compute_features(image):
    # convert image to RGB if it is not already
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # compute brightness and contrast for each channel
    b_brightness = np.mean(image[:, :, 0])
    g_brightness = np.mean(image[:, :, 1])
    r_brightness = np.mean(image[:, :, 2])
    b_contrast = np.std(image[:, :, 0])
    g_contrast = np.std(image[:, :, 1])
    r_contrast = np.std(image[:, :, 2])
    luminance = 0.2126 * r_brightness + 0.7152 * g_brightness + 0.0722 * b_brightness
    
    # compute edge density
    edges = cv2.Canny(image, 100, 200)
    edge_density = np.sum(edges) / (image.shape[0] * image.shape[1])
    
    # compute symmetry
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h_symmetry = np.mean(np.abs(gray_image - np.fliplr(gray_image)))
    v_symmetry = np.mean(np.abs(gray_image - np.flipud(gray_image)))
    
    # compute gradient features
    gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    mean_grad_x = np.mean(gx)
    std_grad_x = np.std(gx)
    mean_grad_y = np.mean(gy)
    std_grad_y = np.std(gy)
    
    # compute shape features
    _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_count = len(contours)
    if object_count > 0:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        bbox_area = w * h
        bbox_ar = w / h
        perimeter = cv2.arcLength(largest, True)
    else:
        x, y, w, h, bbox_area, bbox_ar, perimeter = 0, 0, 0, 0, 0, 0, 0
    
    # compute statistical features
    mean_intensity = np.mean(gray_image)
    variance_intensity = np.var(gray_image)
    entropy = shannon_entropy(gray_image)
    hist = np.histogram(gray_image, bins=256, range=(0, 256))
    mode_intensity = hist[1][np.argmax(hist[0])]
    median_intensity = np.median(gray_image)
    quantiles_intensity = np.percentile(gray_image, [25, 50, 75])
    quantiles_intensity_list = quantiles_intensity.tolist()
    
    # compute frequency domain features
    f_transform = fftshift(fft2(gray_image))
    mag = np.abs(f_transform)
    mean_magnitude = np.mean(mag)
    variance_magnitude = np.var(mag)
    
    # compute structural features
    h, theta, d = hough_line(gray_image)
    _, angles, dists = hough_line_peaks(h, theta, d)
    line_length = np.sum(np.sqrt(np.diff(dists)**2 + np.diff(angles)**2)) if len(dists) > 1 else 0
    
    # compute region-based features
    blobs = blob_log(gray_image, max_sigma=30, num_sigma=10, threshold=.1)
    blob_count = len(blobs)
    grad = sobel(gray_image)
    markers = np.zeros_like(gray_image)
    markers[gray_image < 30] = 1
    markers[gray_image > 150] = 2
    seg = watershed(grad, markers)
    region_homogeneity = np.mean([np.std(gray_image[seg == label]) for label in np.unique(seg)])
    
    # compute noise features
    noise_level = estimate_sigma(gray_image)
    signal_to_noise_ratio = np.mean(gray_image) / noise_level if noise_level != 0 else 0
    
    return {
        'num_pixels_b': np.sum(image[:, :, 0]),
        'num_pixels_g': np.sum(image[:, :, 1]),
        'num_pixels_r': np.sum(image[:, :, 2]),
        'brightness_b': b_brightness,
        'brightness_g': g_brightness,
        'brightness_r': r_brightness,
        'contrast_b': b_contrast,
        'contrast_g': g_contrast,
        'contrast_r': r_contrast,
        'luminance': luminance,
        'edge_density': edge_density,
        'h_symmetry': h_symmetry,
        'v_symmetry': v_symmetry,
        'mean_grad_x': mean_grad_x,
        'std_grad_x': std_grad_x,
        'mean_grad_y': mean_grad_y,
        'std_grad_y': std_grad_y,
        'object_count': object_count,
        'bounding_box_area': bbox_area,
        'bounding_box_aspect_ratio': bbox_ar,
        'bounding_box_x': x,
        'bounding_box_y': y,
        'bounding_box_width': w,
        'bounding_box_height': h,
        'perimeter': perimeter,
        'mean_intensity': mean_intensity,
        'variance_intensity': variance_intensity,
        'entropy': entropy,
        'mode_intensity': mode_intensity,
        'median_intensity': median_intensity,
        # 'quantiles_intensity': quantiles_intensity.tolist(), APPENDED TO END OF DICT !
        'mean_magnitude': mean_magnitude,
        'variance_magnitude': variance_magnitude,
        'line_length': line_length,
        'blob_count': blob_count,
        'region_homogeneity': region_homogeneity,
        'noise_level': noise_level,
        'signal_to_noise_ratio': signal_to_noise_ratio,
        'quantiles_intensity_25': quantiles_intensity_list[0],
        'quantiles_intensity_50': quantiles_intensity_list[1],
        'quantiles_intensity_75': quantiles_intensity_list[2],
    }

# extract pixel values from images using CUDA and batching
def extract_pixel_values_cuda_batched(resized_images, batch_size=32):
    # check if CUDA is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('set ', device, ' as torch.device.')

    pixel_data = []
    print('extracting image pixels by categories:')
    for category, images in resized_images.items():
        num_images = len(images)
        for i in tqdm(range(0, num_images, batch_size), desc=f"    {category}"):
            batch_images = images[i:i+batch_size]
            # Convert batch of images to a single numpy array
            batch_array = np.array(batch_images)
            # Convert numpy array to tensor and move to GPU
            batch_tensor = torch.tensor(batch_array, device=device)
            # Flatten the batch tensor and convert to list
            batch_pixel_values = batch_tensor.view(batch_tensor.size(0), -1).cpu().tolist()
            pixel_data.extend(batch_pixel_values)
    
    # Create a DataFrame with pixel values
    columns = [f'pixel_{i+1}' for i in range(len(pixel_data[0]))]
    print('creating DataFrame from pixel data. This may take a while...')
    pixel_df = pd.DataFrame(pixel_data, columns=columns)
    print('pixel_df successfully created.')
    
    return pixel_df

# assemble and save the final dataset, also save intermediate datasets
def assemble_mvtec_dataset_train_test(
        path_to_mvtec_dataset_dir, 
        force_pkl_overwrite=False, 
        force_pkl_overwrite_resized=False, 
        force_pkl_overwrite_metadata=False, 
        force_pkl_overwrite_features=False, 
        force_pkl_overwrite_pixel=False, 
        force_pkl_overwrite_train_test=False):
    """
    computes and/or loads several datasets and lists from the mvtec anomaly detection dataset. 
    the data is read from .pkl file if it has been previously computed, unless re-computation and overwrite is forced by setting the corresponding flag.

    returns 'categories', 'image_paths', 'resized_images', 'metadata_df', 'features_df', 'pixel_df', 'train_test_df' where 
        categories -- list of categories in mvtec dataset
        image_paths -- dictionary (keys: categories. values: list of paths to all image files for given category)
        resized_images -- dictionary (keys: categories. values: list of resized images as np.arrays for given category)
        metadata_df -- DataFrame containing meatadata for all images
        features_df -- DataFrame containing features computed for all images in resized_images
        pixel_df -- DataFrame with all images from resized_images flattened from (224 x 224 x 3) to (1 x 150528) 
        train_test_df -- DataFrame containing metadata_df, features_df and pixel_df concatenated into one

    creates folder 'path_to_mvtec_dataset_dir/pickle_files' and saves the following data as pickle files (.pkl) if not present or if forced:
        resized_images to 'resized_images.pkl'
        metadata_df to 'image_metadata.pkl'
        features_df to 'image_features.pkl'
        pixel_df to 'image_pixel.pkl'
        train_test_df to 'train_test_df_metadata_features_pixels.pkl'
    
    Arguments:
        path_to_mvtec_dataset_dir -- path to mvtec_anomaly_detection dataset (the folder, that contains all the category folders)
        force_pkl_overwrite -- re-compute all extracted data, and overwrite existing .pkl files
        force_pkl_overwrite_resized -- re-compute resized_images and overwrite existing .pkl file
        force_pkl_overwrite_metadata -- re-compute metadata_df and overwrite existing .pkl file
        force_pkl_overwrite_features -- re-compute features_df and overwrite existing .pkl file
        force_pkl_overwrite_pixel -- re-compute pixel_df and overwrite existing .pkl file
        force_pkl_overwrite_train_test -- re-compute train_test_df and overwrite existing .pkl file
    """

    # check, if path is valid
    if not os.path.exists(path_to_mvtec_dataset_dir):
        print('given path is not valid!')
        return None
    
    # folder to save intermediate results to .pkl files
    pkl_path = os.path.join(path_to_mvtec_dataset_dir, 'pickle_files')
    if not os.path.exists(pkl_path):
        os.mkdir(pkl_path)
    
    # get list of all categories
    categories = get_category_list(path_to_mvtec_dataset_dir)
    print('category names have been extracted.')

    # get all image and ground_truth paths for all categories
    train_image_paths = {} # for each category, list of all paths to the images in the 'train' subfolder
    test_image_paths = {} # for each category, list of all paths to the images in the 'test' subfolder
    image_paths = {} # for each category, list of all paths to the images in the 'train' and 'test' subfolders, combined (i.e. to all images except ground_truth)
    ground_truth_paths = {} # for each category, list of all paths to the images in the 'ground_truth' subfolder
    
    for category in categories:
        train_image_paths[category], test_image_paths[category], ground_truth_paths[category], image_paths[category] = get_all_image_and_ground_truth_paths(path_to_mvtec_dataset_dir, category)
    print('image paths have been extracted.')
    
    ######################## RESIZED IMAGES DICT ########################
    # load or create resized_images
    if not os.path.exists(os.path.join(pkl_path, 'resized_images.pkl')) or force_pkl_overwrite or force_pkl_overwrite_resized:
        # resize all images from 'train' and 'test' and save to .pkl file
        print('resizing images by categories:')
        resized_images = resize_images(image_paths)
        print('images have been resized.')
        with open(os.path.join(pkl_path, 'resized_images.pkl'), 'wb') as f:
            pickle.dump(resized_images, f)
        print('resized images saved to pickle as ', os.path.join(pkl_path, 'resized_images.pkl'), '.')
    else:
        with open(os.path.join(pkl_path, 'resized_images.pkl'), 'rb') as f:
            resized_images = pickle.load(f)
        print('loaded resized_images from .pkl file.')
    
    ######################## METADATA DATAFRAME ########################
    # load or create metadata_df
    if not os.path.exists(os.path.join(pkl_path, 'image_metadata.pkl')) or force_pkl_overwrite or force_pkl_overwrite_metadata:
        # extract metadata for images in 'train' and 'test', add subclasses and save to .pkl file
        print('extracting image metadata by categories:')
        image_metadata = extract_image_metadata(image_paths)
        # convert metadata to pandas DataFrame
        metadata_df = pd.DataFrame(image_metadata)
        # define subclasses for each category
        subclasses = {
            'Texture-Based': ['carpet', 'wood', 'tile', 'leather', 'zipper'],
            'Industrial Components': ['cable', 'transistor', 'screw', 'grid', 'metal_nut'],
            'Consumer Products': ['bottle', 'capsule', 'toothbrush'],
            'Edible': ['hazelnut', 'pill']
        }
        # add new column to DataFrame for subclasses
        metadata_df['subclass'] = metadata_df['category'].apply(
            lambda x: next((key for key, value in subclasses.items() if x in value), 'Unknown')
        )
        # reorder columns to place 'subclass' after 'category'
        cols = list(metadata_df.columns)
        cols.insert(cols.index('category') + 1, cols.pop(cols.index('subclass')))
        metadata_df = metadata_df[cols]
        # save metadata to pkl file
        metadata_df.to_pickle(os.path.join(pkl_path, 'image_metadata.pkl'))
        print('image metadata saved to pickle as ', os.path.join(pkl_path, 'image_metadata.pkl', '.'))
    else:
        metadata_df = pd.read_pickle(os.path.join(pkl_path, 'image_metadata.pkl'))
        print('loaded metadata_df from .pkl file.')

    ######################## FEATURES DATAFRAME ########################
    # load or create features_df
    if not os.path.exists(os.path.join(pkl_path, 'image_features.pkl')) or force_pkl_overwrite or force_pkl_overwrite_features:
        # extract image features from images in 'train' and 'test' and save to .pkl file
        image_features = []
        print('extracting image features by categories:')
        for category, images in resized_images.items():
            for image in tqdm(images, desc=f"    {category}"):
                features = compute_features(image)
                image_features.append(features)
        # convert features to DataFrame
        features_df = pd.DataFrame(image_features)
        # save features to pkl file
        features_df.to_pickle(os.path.join(pkl_path, 'image_features.pkl'))
        print('image features saved to pickle as ', os.path.join(pkl_path, 'image_features.pkl', '.'))
    else:
        features_df = pd.read_pickle(os.path.join(pkl_path, 'image_features.pkl'))
        print('loaded features_df from .pkl file.')
    
    ######################## PIXEL DATAFRAME ########################
    # load or create pixel_df
    if not os.path.exists(os.path.join(pkl_path, 'image_pixel.pkl')) or force_pkl_overwrite or force_pkl_overwrite_pixel:
        # extract pixel values from images in 'train' and 'test', create DataFrame and save to .pkl file
        pixel_df = extract_pixel_values_cuda_batched(resized_images)
        # save pixel_df to pkl file
        pixel_df.to_pickle(os.path.join(pkl_path, 'image_pixel.pkl'))
        print('pixel_df saved to pickle as ', os.path.join(pkl_path, 'image_pixel.pkl', '.'))
    else:
        pixel_df = pd.read_pickle(os.path.join(pkl_path, 'image_pixel.pkl'))
        print('loaded pixel_df from .pkl file.')
    
    ######################## COMPLETE TRAIN TEST DATAFRAME ########################
    # load or create train_test_df
    if not os.path.exists(os.path.join(pkl_path, 'train_test_df_metadata_features_pixels.pkl')) or force_pkl_overwrite or force_pkl_overwrite_train_test:
        # concatenate metadata, features and pixel dataframes along columns and save to .pkl file
        train_test_df = pd.concat([metadata_df, features_df, pixel_df], axis=1)
        # save train_test_df to pkl file
        train_test_df.to_pickle(os.path.join(pkl_path, 'train_test_df_metadata_features_pixels.pkl'))
        print('train_test_df saved to pickle as ', os.path.join(pkl_path, 'train_test_df_metadata_features_pixels.pkl', '.'))
    else:
        train_test_df = pd.read_pickle(os.path.join(pkl_path, 'train_test_df_metadata_features_pixels.pkl'))
        print('loaded train_test_df from .pkl file.')

    return categories, image_paths, resized_images, metadata_df, features_df, pixel_df, train_test_df

