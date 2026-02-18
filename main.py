"""
======================================================================
FACE RECOGNITION USING PCA AND ANN (ARTIFICIAL NEURAL NETWORK)
======================================================================
Internship Project: Implementation of PCA with ANN Algorithm for Face Recognition

Author: Student
Date: 2026
Description: 
    This program implements Principal Component Analysis (PCA) for face 
    dimensionality reduction and Artificial Neural Network (ANN) for 
    face classification. It does manual PCA without using sklearn's PCA.
    
Dependencies:
    - numpy: Numerical computations
    - opencv-python (cv2): Image processing
    - matplotlib: Visualization
    - scikit-learn: Train/test split and ANN classifier
    
======================================================================
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better compatibility
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# ======================================================================
# CONFIGURATION SECTION
# ======================================================================

# Image preprocessing configuration
IMG_WIDTH = 100          # Resize images to 100x100 pixels
IMG_HEIGHT = 100
DATASET_FOLDER = "Dataset/faces"  # Folder containing Aamir/, Ajay/, etc.

# PCA configuration
K_VALUES = [5, 10, 20, 30, 40]  # Different k values to test
K_FINAL = 90             # Final k for main model

# Train-test split configuration
TRAIN_SIZE = 0.8         # 60% training
TEST_SIZE = 0.2          # 40% testing

# ANN configuration
RANDOM_STATE = 42        # For reproducibility
HIDDEN_LAYER_SIZE = (32,)  # ANN architecture: 128 neurons, then 64 neurons
MAX_ITER = 3000          # Maximum iterations for ANN training

# Imposter detection configuration
IMPOSTER_THRESHOLD = None  # None = auto-calibrate threshold from training data
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".pgm"}

print("=" * 70)
print("FACE RECOGNITION USING PCA AND ANN".center(70))
print("=" * 70)
print()


# ======================================================================
# STEP 1: LOAD AND PREPROCESS IMAGES
# ======================================================================

def load_and_preprocess_images():
    """
    Load images from dataset folder, convert to grayscale, 
    resize to 100x100, and flatten to column vectors.
    Dataset Structure Expected:
        Dataset/
        ├── faces/
        │   ├── img1.jpg
        │   ├── img2.jpg
        ├── iris/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── ...
    
    Returns:
        Face_Db (numpy array): Shape (10000, num_images) - flattened face matrix
        labels (numpy array): Shape (num_images,) - class labels (0, 1, 2, ...)
        label_names (list): List of person names (folder names)
    """
    
    print("STEP 1: LOADING AND PREPROCESSING IMAGES")
    print("-" * 70)
    
    if not os.path.exists(DATASET_FOLDER):
        print(f"ERROR: '{DATASET_FOLDER}' folder not found!")
        print(f"Please create the folder structure:")
        print(f"  {DATASET_FOLDER}/")
        print(f"  ├── person1/")
        print(f"  │   ├── img1.jpg")
        print(f"  │   └── img2.jpg")
        print(f"  └── person2/")
        print(f"      ├── img1.jpg")
        print(f"      └── img2.jpg")
        return None, None, None
    
    Face_Db = []           # Will store flattened images
    labels = []            # Will store person labels
    label_names = []       # Will store person names
    label_id = 0
    
    # Iterate through each person folder
    for person_name in sorted(os.listdir(DATASET_FOLDER)):
        person_path = os.path.join(DATASET_FOLDER, person_name)
        
        # Skip if not a directory
        if not os.path.isdir(person_path):
            continue
        
        print(f"Loading images from: {person_name}")

        # Load all images from person's folder
        image_count = 0
        person_images = []
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            # Skip nested folders or non-image files
            if os.path.isdir(img_path):
                continue
            _, ext = os.path.splitext(img_name)
            if ext.lower() not in VALID_IMAGE_EXTENSIONS:
                continue
            
            # Read image in grayscale
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"  WARNING: Could not read {img_path}")
                continue
            
            # Resize to standard size (100x100)
            image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            
            # Flatten to column vector (10000,)
            image_flattened = image_resized.flatten()
            
            # Keep person images local; assign class only if at least one image is valid
            person_images.append(image_flattened)
            image_count += 1
        
        if image_count == 0:
            print(f"  WARNING: No valid images found for {person_name}. Skipping class.")
            continue

        label_names.append(person_name)
        Face_Db.extend(person_images)
        labels.extend([label_id] * image_count)
        print(f"  [OK] Loaded {image_count} images for {person_name}")
        label_id += 1
    
    if len(Face_Db) == 0:
        print("ERROR: No images loaded! Check dataset folder structure.")
        return None, None, None
    
    # Convert to numpy arrays
    Face_Db = np.array(Face_Db).T  # Shape: (10000, num_images)
    labels = np.array(labels)
    
    print(f"\n[OK] Data Loading Complete!")
    print(f"  Face_Db shape: {Face_Db.shape}")  # Should be (10000, num_images)
    print(f"  Number of persons: {len(label_names)}")
    print(f"  Total images: {len(labels)}")
    print(f"  Labels: {labels}")
    print(f"  Person names: {label_names}")
    print()
    
    return Face_Db, labels, label_names


# ======================================================================
# STEP 2: COMPUTE MEAN FACE
# ======================================================================

def compute_mean_face(Face_Db):
    """
    Compute the average face from all images.
    This is the mean across all columns (images).
    
    Args:
        Face_Db (numpy array): Shape (10000, num_images)
    
    Returns:
        mean_face (numpy array): Shape (10000,)
    """
    
    print("STEP 2: COMPUTING MEAN FACE")
    print("-" * 70)
    
    # Mean across all images (axis=1 means average across columns)
    mean_face = np.mean(Face_Db, axis=1)
    
    print(f"[OK] Mean face computed!")
    print(f"  Mean face shape: {mean_face.shape}")  # Should be (10000,)
    print()
    
    return mean_face


# ======================================================================
# STEP 3: MEAN CENTER THE DATA (Subtract mean from each image)
# ======================================================================

def mean_center_data(Face_Db, mean_face):
    """
    Subtract mean face from all images.
    This centers the data around zero.
    
    Mathematical Operation: Δ = Face_Db - mean_face (broadcast subtraction)
    
    Args:
        Face_Db (numpy array): Shape (10000, num_images)
        mean_face (numpy array): Shape (10000,)
    
    Returns:
        Delta (numpy array): Mean-centered data, shape (10000, num_images)
    """
    
    print("STEP 3: MEAN CENTERING DATA")
    print("-" * 70)
    
    # Broadcast: subtract mean_face from each column
    Delta = Face_Db - mean_face.reshape(-1, 1)
    
    print(f"[OK] Data mean-centered!")
    print(f"  Delta shape: {Delta.shape}")  # Should be (10000, num_images)
    print()
    
    return Delta


# ======================================================================
# STEP 4: COMPUTE SURROGATE COVARIANCE MATRIX
# ======================================================================

def compute_covariance_matrix(Delta):
    """
    Compute covariance matrix using the surrogate method.
    
    Instead of computing: C = (1/m) * Δ * Δ^T  (size: 10000 x 10000)
    We compute: L = (1/m) * Δ^T * Δ            (size: num_images x num_images)
    
    Then eigenvalues of C = eigenvalues of L
    And eigenvectors of C can be computed from eigenvectors of L
    
    This is much more efficient when num_images << image_dimension
    
    Args:
        Delta (numpy array): Shape (10000, num_images)
    
    Returns:
        L (numpy array): Surrogate covariance matrix, shape (num_images, num_images)
    """
    
    print("STEP 4: COMPUTING SURROGATE COVARIANCE MATRIX")
    print("-" * 70)
    
    # Number of images
    m = Delta.shape[1]
    
    # Surrogate covariance: L = (1/m) * Δ^T * Δ
    L = (1 / m) * np.dot(Delta.T, Delta)
    
    print(f"[OK] Covariance matrix computed!")
    print(f"  L (Surrogate covariance) shape: {L.shape}")  # Should be (num_images, num_images)
    print(f"  L condition number: {np.linalg.cond(L):.2e}")
    print()
    
    return L


# ======================================================================
# STEP 5: EIGEN DECOMPOSITION
# ======================================================================

def eigen_decomposition(L):
    """
    Perform eigen decomposition on the surrogate covariance matrix.
    
    This finds eigenvalues and eigenvectors of L.
    
    Args:
        L (numpy array): Surrogate covariance matrix, shape (num_images, num_images)
    
    Returns:
        eigenvalues (numpy array): Eigenvalues of L (real)
        eigenvectors (numpy array): Eigenvectors of L (columns are eigenvectors, real)
    """
    
    print("STEP 5: EIGEN DECOMPOSITION")
    print("-" * 70)
    
    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(L)
    
    # Extract real parts (imaginary parts are negligible numerical errors)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    print(f"[OK] Eigen decomposition complete!")
    print(f"  Eigenvalues shape: {eigenvalues.shape}")
    print(f"  Eigenvectors shape: {eigenvectors.shape}")
    print(f"  Top 5 eigenvalues: {eigenvalues[:5]}")
    print()
    
    return eigenvalues, eigenvectors


# ======================================================================
# STEP 6: SORT EIGENVALUES IN DESCENDING ORDER
# ======================================================================

def sort_eigenvalues(eigenvalues, eigenvectors):
    """
    Sort eigenvalues in descending order and reorder eigenvectors accordingly.
    
    The principal components are those with largest eigenvalues.
    
    Args:
        eigenvalues (numpy array): Eigenvalues from eigen decomposition
        eigenvectors (numpy array): Eigenvectors from eigen decomposition
    
    Returns:
        sorted_eigenvalues (numpy array): Eigenvalues in descending order
        sorted_eigenvectors (numpy array): Eigenvectors reordered
    """
    
    print("STEP 6: SORTING EIGENVALUES")
    print("-" * 70)
    
    # Get indices that would sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    
    # Reorder
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]
    
    print(f"[OK] Eigenvalues sorted in descending order!")
    print(f"  Eigenvalues (top 10): {sorted_eigenvalues[:10]}")
    print()
    
    return sorted_eigenvalues, sorted_eigenvectors


# ======================================================================
# STEP 7: SELECT K PRINCIPAL COMPONENTS
# ======================================================================

def select_k_components(sorted_eigenvalues, sorted_eigenvectors, k):
    """
    Select the top k eigenvectors (principal components) based on largest eigenvalues.
    
    Args:
        sorted_eigenvalues (numpy array): Sorted eigenvalues (descending)
        sorted_eigenvectors (numpy array): Sorted eigenvectors
        k (int): Number of components to select
    
    Returns:
        selected_eigenvectors (numpy array): Top k eigenvectors, shape (num_images, k)
    """
    
    print(f"STEP 7: SELECTING K={k} PRINCIPAL COMPONENTS")
    print("-" * 70)
    
    max_k = sorted_eigenvectors.shape[1]
    if k > max_k:
        print(f"  WARNING: Requested k={k} exceeds max={max_k}. Using k={max_k}.")
        k = max_k
    if k < 1:
        raise ValueError("k must be >= 1")

    # Select top k eigenvectors
    selected_eigenvectors = sorted_eigenvectors[:, :k]
    
    print(f"[OK] Selected top {k} components!")
    print(f"  Selected eigenvectors shape: {selected_eigenvectors.shape}")
    print()
    
    return selected_eigenvectors


# ======================================================================
# STEP 8: GENERATE EIGENFACES
# ======================================================================

def generate_eigenfaces(Delta, selected_eigenvectors):
    """
    Generate eigenfaces by projecting selected eigenvectors back to image space.
    
    Eigenfaces = Δ * selected_eigenvectors
    
    Note: The eigenvectors from L are in the reduced space.
    To get them back in the original image space, multiply with Δ.
    
    Args:
        Delta (numpy array): Mean-centered data, shape (10000, num_images)
        selected_eigenvectors (numpy array): Selected eigenvectors, shape (num_images, k)
    
    Returns:
        eigenfaces (numpy array): Eigenfaces in image space, shape (10000, k)
    """
    
    print("STEP 8: GENERATING EIGENFACES")
    print("-" * 70)
    
    # Project to original space: each eigenface is a linear combination of centered images
    eigenfaces = np.dot(Delta, selected_eigenvectors)
    
    # Ensure real values
    eigenfaces = np.real(eigenfaces)
    
    # Normalize each eigenface to unit norm for stable projection scales
    norms = np.linalg.norm(eigenfaces, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    eigenfaces = eigenfaces / norms
    
    print(f"[OK] Eigenfaces generated!")
    print(f"  Eigenfaces shape: {eigenfaces.shape}")  # Should be (10000, k)
    print()
    
    return eigenfaces


# ======================================================================
# STEP 9: GENERATE SIGNATURES (PROJECT DATA ONTO EIGENFACES)
# ======================================================================

def generate_signatures(Face_Db, mean_face, eigenfaces):
    """
    Project all face images onto the eigenfaces to generate signatures.
    
    Signature for image i = eigenfaces^T * (image_i - mean_face)
    
    This transforms high-dimensional image data to low-dimensional PCA space.
    
    Args:
        Face_Db (numpy array): Original face data, shape (10000, num_images)
        mean_face (numpy array): Mean face, shape (10000,)
        eigenfaces (numpy array): Eigenfaces, shape (10000, k)
    
    Returns:
        signatures (numpy array): PCA signatures, shape (num_images, k)
    """
    
    print("STEP 9: GENERATING SIGNATURES (PCA PROJECTION)")
    print("-" * 70)
    
    # Mean-center each image
    Delta = Face_Db - mean_face.reshape(-1, 1)
    
    # Project onto eigenfaces: signatures = eigenfaces^T * Delta
    signatures = np.dot(eigenfaces.T, Delta).T  # Shape: (num_images, k)
    
    # Ensure real values
    signatures = (signatures - np.mean(signatures,axis=0))/(np.std(signatures,axis=0)+1e-8)  # Standardize signatures
    
    print(f"[OK] Signatures generated!")
    print(f"  Signatures shape: {signatures.shape}")  # Should be (num_images, k)
    print()
    
    return signatures


# ======================================================================
# STEP 10: SPLIT DATASET INTO TRAINING AND TESTING
# ======================================================================

def split_dataset(signatures, labels):
    """
    Split signatures and labels into training and testing sets.
    
    60% training, 40% testing
    
    Args:
        signatures (numpy array): PCA signatures, shape (num_images, k)
        labels (numpy array): Class labels, shape (num_images,)
    
    Returns:
        X_train, X_test, y_train, y_test: Training and testing sets
    """
    
    print("STEP 10: SPLITTING DATASET")
    print("-" * 70)
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    can_stratify = (
        len(unique_labels) > 1 and
        np.all(counts >= 2) and
        len(labels) * TEST_SIZE >= len(unique_labels)
    )
    stratify_labels = labels if can_stratify else None
    if not can_stratify:
        print("  WARNING: Stratified split not possible with current label counts. Using random split.")

    X_train, X_test, y_train, y_test = train_test_split(
        signatures, labels,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_labels
    )

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # ------------------------

    return X_train, X_test, y_train, y_test
    
    print(f"[OK] Dataset split complete!")
    print(f"  Training set size: {X_train.shape[0]} ({TRAIN_SIZE*100:.0f}%)")
    print(f"  Testing set size: {X_test.shape[0]} ({TEST_SIZE*100:.0f}%)")
    print(f"  X_train shape: {X_train.shape}")  # Should be (num_train, k)
    print(f"  X_test shape: {X_test.shape}")    # Should be (num_test, k)
    print()
    
    return X_train, X_test, y_train, y_test


# ======================================================================
# STEP 11: TRAIN ARTIFICIAL NEURAL NETWORK (ANN)
# ======================================================================

def train_ann(X_train, y_train):
    """
    Train an Artificial Neural Network using MLPClassifier from scikit-learn.
    
    Architecture: Input layer (k neurons) -> Hidden layer 1 (128 neurons) 
                  -> Hidden layer 2 (64 neurons) -> Output layer (num_classes neurons)
    
    Args:
        X_train (numpy array): Training signatures, shape (num_train, k)
        y_train (numpy array): Training labels, shape (num_train,)
    
    Returns:
        ann_model: Trained MLPClassifier model
    """
    
    print("STEP 11: TRAINING ARTIFICIAL NEURAL NETWORK (ANN)")
    print("-" * 70)
    
    # Create ANN model
    # Architecture: 128 neurons in first hidden layer, 64 in second
    ann_model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZE,
        activation='relu',          # ReLU activation
        solver='adam',              # Adam optimizer
        learning_rate='adaptive',
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    print(f"Training ANN with architecture: {HIDDEN_LAYER_SIZE}")
    print(f"Solver: adam | Activation: relu | Max iterations: {MAX_ITER}")
    
    # Train the model
    ann_model.fit(X_train, y_train)
    
    print(f"\n[OK] ANN training complete!")
    print(f"  Training accuracy: {ann_model.score(X_train, y_train):.4f}")
    print()
    
    return ann_model


# ======================================================================
# STEP 12: EVALUATE MODEL (ACCURACY AND CONFUSION MATRIX)
# ======================================================================

def evaluate_model(ann_model, X_test, y_test, label_names):
    """
    Evaluate the trained ANN on test data.
    
    Computes accuracy and confusion matrix.
    
    Args:
        ann_model: Trained ANN model
        X_test (numpy array): Test signatures
        y_test (numpy array): Test labels
        label_names (list): Person names for confusion matrix
    
    Returns:
        accuracy (float): Test accuracy
        conf_matrix (numpy array): Confusion matrix
    """
    
    print("STEP 12: EVALUATING MODEL")
    print("-" * 70)
    
    # Make predictions
    y_pred = ann_model.predict(X_test)
    
    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(label_names))
)

    
    print(f"[OK] Model evaluated!")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    
    print("Confusion Matrix:")
    print("-" * 70)
    
    # Print confusion matrix with labels
    header = "Actual \\ Predicted"
    print(f"{header:<20}", end="")
    for label_name in label_names:
        print(f"{label_name:<15}", end="")
    print()
    
    for i, label_name in enumerate(label_names):
        print(f"{label_name:<20}", end="")
        for j in range(len(label_names)):
            print(f"{conf_matrix[i, j]:<15}", end="")
        print()
    
    print()
    
    return accuracy, conf_matrix


# ======================================================================
# STEP 13: ACCURACY VS K EXPERIMENT
# ======================================================================

def accuracy_vs_k_experiment(Face_Db, labels, label_names, mean_face, Delta, 
                             sorted_eigenvalues, sorted_eigenvectors):
    """
    Test different k values (number of principal components) and plot accuracy.
    
    This shows how many components are needed for good performance.
    
    Args:
        Face_Db (numpy array): Original face data
        labels (numpy array): Face labels
        label_names (list): Person names
        mean_face (numpy array): Mean face
        Delta (numpy array): Mean-centered data
        sorted_eigenvalues (numpy array): Sorted eigenvalues
        sorted_eigenvectors (numpy array): Sorted eigenvectors
    
    Returns:
        accuracies (list): Accuracy for each k value
    """
    
    print("STEP 13: ACCURACY VS K EXPERIMENT")
    print("-" * 70)
    print(f"Testing k values: {K_VALUES}")
    print()
    
    accuracies = []
    
    for k in K_VALUES:
        # Select k components
        selected_eigenvectors = select_k_components(sorted_eigenvalues, 
                                                     sorted_eigenvectors, k)
        
        # Generate eigenfaces
        eigenfaces = generate_eigenfaces(Delta, selected_eigenvectors)
        
        # Generate signatures
        signatures = generate_signatures(Face_Db, mean_face, eigenfaces)
        
        # Split dataset
        X_train, X_test, y_train, y_test = split_dataset(signatures, labels)
        
        # Train ANN
        ann_model = train_ann(X_train, y_train)
        
        # Evaluate
        accuracy, _ = evaluate_model(ann_model, X_test, y_test, label_names)
        accuracies.append(accuracy)
        
        print(f"k={k}: Accuracy = {accuracy:.4f}")
    
    print()
    
    # Plot accuracy vs k
    plt.figure(figsize=(10, 6))
    plt.plot(K_VALUES, accuracies, marker='o', linewidth=2, markersize=8, color='blue')
    plt.xlabel('Number of Principal Components (k)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy vs Number of Principal Components (PCA)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(K_VALUES)
    plt.ylim([0, 1.05])
    
    # Add value labels on points
    for k, acc in zip(K_VALUES, accuracies):
        plt.text(k, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('accuracy_vs_k.png', dpi=300, bbox_inches='tight')
    print("[OK] Graph saved as 'accuracy_vs_k.png'")
    
    # Try to show graph (won't work in all environments, but won't crash)
    try:
        import matplotlib
        if matplotlib.get_backend() != 'agg':
            plt.show()
    except:
        pass  # Silent fail - graph is already saved
    
    plt.close()  # Close to free memory
    print()
    
    return accuracies


# ======================================================================
# STEP 14: IMPOSTER DETECTION
# ======================================================================

# STEP 14: Imposter detection
def detect_imposters(
    X_test,
    X_train,
    y_train,
    ann_model,
    label_names
):


    """
    Detect if a test face belongs to an unknown person (imposter).

    Uses distance threshold logic in PCA signature space.
    """

    print("STEP 14: IMPOSTER DETECTION")
    print("-" * 70)

    # Get ANN predictions
    y_pred = ann_model.predict(X_test)

    imposter_count = 0
    enrolled_count = 0

    # Auto-calibrate threshold from class-wise nearest-neighbor distances in training set
    if IMPOSTER_THRESHOLD is None:
        nn_distances = []
        for label in np.unique(y_train):
            class_sigs = X_train[y_train == label]
            if len(class_sigs) < 2:
                continue
            dmat = np.linalg.norm(class_sigs[:, None, :] - class_sigs[None, :, :], axis=2)
            np.fill_diagonal(dmat, np.inf)
            nn_distances.extend(np.min(dmat, axis=1))
        threshold = float(np.percentile(nn_distances, 95)) if nn_distances else float("inf")
    else:
        threshold = float(IMPOSTER_THRESHOLD)

    print(f"Testing {len(X_test)} images for imposter detection:")
    print("(Using euclidean distance threshold = {:.4f})".format(threshold))
    print()

    for idx in range(min(10, len(X_test))):  # Show first 10 results

        test_sig = X_test[idx]
        pred_label = y_pred[idx]

        # Compute distance from test signature to training signatures of predicted class
        class_sigs = X_train[y_train == pred_label]

        if len(class_sigs) > 0:
            distances = np.linalg.norm(class_sigs - test_sig, axis=1)
            min_distance = np.min(distances)
        else:
            min_distance = float('inf')

        # Imposter decision
        if min_distance > threshold:
            status = "NOT ENROLLED (Imposter)"
            imposter_count += 1
        else:
            status = "ENROLLED"
            enrolled_count += 1

        print(f"Image {idx+1}: Predicted={label_names[pred_label]}, "
              f"Distance={min_distance:.4f}, Status={status}")

    print()
    print(f"Summary of first 10 test images:")
    print(f"  Enrolled persons: {enrolled_count}")
    print(f"  Imposter (Not Enrolled): {imposter_count}")
    print()


# ======================================================================
# MAIN FUNCTION - ORCHESTRATE ALL STEPS
# ======================================================================

def main():
    """
    Main function that orchestrates the entire face recognition pipeline.
    """
    
    # STEP 1: Load and preprocess images
    Face_Db, labels, label_names = load_and_preprocess_images()
    
    if Face_Db is None:
        print("Failed to load images. Exiting.")
        return
    
    # STEP 2: Compute mean face
    mean_face = compute_mean_face(Face_Db)
    
    # STEP 3: Mean center the data
    Delta = mean_center_data(Face_Db, mean_face)
    
    # STEP 4: Compute surrogate covariance matrix
    L = compute_covariance_matrix(Delta)
    
    # STEP 5: Eigen decomposition
    eigenvalues, eigenvectors = eigen_decomposition(L)
    
    # STEP 6: Sort eigenvalues
    sorted_eigenvalues, sorted_eigenvectors = sort_eigenvalues(eigenvalues, eigenvectors)
    
    # STEP 7: Select k principal components (for final model)
    selected_eigenvectors = select_k_components(sorted_eigenvalues, 
                                                sorted_eigenvectors, K_FINAL)
    
    # STEP 8: Generate eigenfaces
    eigenfaces = generate_eigenfaces(Delta, selected_eigenvectors)
    
    # STEP 9: Generate signatures
    signatures = generate_signatures(Face_Db, mean_face, eigenfaces)
    
    # STEP 10: Split dataset
    X_train, X_test, y_train, y_test = split_dataset(signatures, labels)
    
    # STEP 11: Train ANN
    ann_model = train_ann(X_train, y_train)
    
    # STEP 12: Evaluate model
    accuracy, conf_matrix = evaluate_model(ann_model, X_test, y_test, label_names)
    
    print("=" * 70)
    print("MAIN FACE RECOGNITION MODEL COMPLETE")
    print(f"Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 70)
    print()
    
    # STEP 13: Accuracy vs k experiment
    accuracy_vs_k_experiment(Face_Db, labels, label_names, mean_face, Delta,
                             sorted_eigenvalues, sorted_eigenvectors)
    
    # STEP 14: Imposter detection
    detect_imposters(X_test, X_train, y_train, ann_model, label_names)
    
    print("=" * 70)
    print("FACE RECOGNITION PROJECT COMPLETE!")
    print("=" * 70)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()
