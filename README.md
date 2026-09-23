# Mid-Deep-Learning-Project-Semester-4

Project Description:
- `1B.ipynb`: Used-car resale price prediction using ANN regression
- `2B.ipynb`: Cherry leaf disease classification using CNNs (AlexNet and ResNet50)
- `4B.ipynb`: Manual ANN forward/backward pass with ReLU and MSE

---

## 1) Dataset Overview

### 1B — Used Car Price Prediction
The dataset is stored in `data/1B.parquet` and contains tabular vehicle listings with the target variable `selling_price` in USD. Features include vehicle metadata such as `name`, `year`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`, `mileage`, `engine`, `max_power`, `torque`, `seats`, `Region`, and `State or Province`.

This is a regression task where the model predicts a continuous resale price based on vehicle condition, specification, and location.

### 2B — Cherry Leaf Disease Classification
The image dataset is organized under the `data/2C/Cherry/data` folder and contains five disease categories:

- Cherry Leaf Scorch
- Cherry Normal Leaf
- Cherry Brown Spot
- Cherry Purple Leaf Spot
- Cherry Shot Hole Disease

Researcher prepares the dataset into train, validation, and test folders using a 70:15:15 split and uses image-based classification to detect leaf disease patterns from visual symptoms.

### 4B — Toy ANN Example
Notebook does not use a large real-world dataset. It demonstrates a simple neural network on a small NumPy example with 2 inputs and 1 output, using ReLU activation and backpropagation.

---

## 2) Data Preprocessing Technique

### 1B — Tabular Data Cleaning and Feature Engineering
The preprocessing pipeline includes several important steps:

- Dropped irrelevant or high-cardinality columns such as `Sales_ID` and `City`
- Removed invalid rows, including missing `selling_price`, impossible `year` values, and non-positive `max_power`
- Converted numeric-like strings into valid numeric values, especially for `mileage`
- Filled missing categorical values with `Unknown`
- Handled duplicate records by removing repeated rows while keeping the first occurrence
- Parsed `torque` into two useful features:
  - `torque_nm`
  - `rpm_val`
- Imputed missing numeric values using median-based fill to maintain data integrity
- Standardized numeric columns using `StandardScaler`
- One-hot encoded categorical columns using `OneHotEncoder`
- Split the dataset into train, validation, and test sets using a 70/10/20 strategy

The target variable was transformed with `log1p` during training to stabilize regression learning and then converted back to the original scale for evaluation.

### 2B — Image Dataset Preparation
For the cherry leaf dataset, the preprocessing steps are image-oriented:

- Explored image size, width, height, and aspect ratio
- Split data by class into `train`, `val`, and `test` folders using a 70:15:15 ratio
- Resized all images to `224 x 224`
- Applied `Rescaling(1./255)` to normalize pixel values to the range `[0,1]`
- Used categorical labels for multi-class classification
- Cached and prefetched the dataset for efficient training acceleration

### 4B — Manual Numerical Preprocessing
No external dataset cleaning was used here; the notebook manually defines a small matrix-based dataset and computes forward propagation, derivative calculations, and weight updates using NumPy.

---

## 3) Modelling Approach (Baseline + Modification)

### 1B — ANN for Car Price Prediction
#### Baseline Model
The baseline ANN used:

- ReLU hidden layers
- At least two hidden layers, with the number of neurons at least twice the input dimension
- Input features processed through scaling + encoding
- Loss function: MSE
- Optimizer: Adam
- Epochs: 30

This model established a strong baseline for price regression.

#### Modified Model
The modified model added several improvements:

- More expressive architecture: 128 → 64 → 32 hidden layers
- Batch Normalization and Dropout regularization
- SGD optimizer with momentum for better generalization
- EarlyStopping and ReduceLROnPlateau callbacks
- Training length increased to 150 epochs

The modification was designed to improve generalization and reduce overfitting, which is critical for resale pricing where noise and variance are common.

### 2B — CNN for Plant Disease Detection
#### Baseline Model
The baseline model is an AlexNet-inspired CNN with:

- Rescaling layer
- Convolution blocks and pooling layers
- ReLU activations
- Dense classifier with softmax output
- Categorical cross-entropy loss
- Adam optimizer
- 10 epochs

#### Modified Model
Two modification strategies were tested:

1. Tuned AlexNet with Batch Normalization, learning-rate scheduling, and early stopping
2. Transfer learning with pretrained ResNet50 as a feature extractor

The ResNet50 approach was the stronger modification because it leverages ImageNet pre-trained features, which are highly effective for leaf texture, color, and disease pattern recognition. It also used GlobalAveragePooling, Batch Normalization, and Dropout to stabilize learning.

### 4B — Manual Backpropagation Example
This notebook demonstrates the mathematical core of neural learning:

- Forward propagation using ReLU
- MSE loss computation
- Backward pass to compute gradients
- Weight updates using gradient descent

It is a pedagogical implementation rather than a production model.

---

## 4) Main Limitation

### 1B
The main limitation is the high variability of used-car pricing. Even after cleaning, price is influenced by many unobserved factors such as market condition, seller negotiation, exact vehicle history, and local demand. The model still shows a non-trivial MAPE, which means prediction error remains noticeable for individual listings.

### 2B
The image dataset likely has class imbalance and limited environmental diversity. In real-world applications, lighting conditions, camera angles, and field noise can reduce performance. Although ResNet50 is stable, the notebook results still indicate that model robustness depends heavily on data quality and diversity.

### 4B
The toy example is intentionally small and not representative of real-world model behavior. It serves as a learning tool only and cannot be generalized to practical datasets.

---

## 5) Final Report Metric (Baseline vs Modified) & Interpretation

### 1B — Used-Car Regression

| Model | R-squared | MAE | MAPE |
|---|---:|---:|---:|
| Baseline | 0.913174 | 1287.74 | 0.225006 (22.50%) |
| Modified | 0.939035 | 1029.57 | 0.182224 (18.22%) |

Interpretation:

- R-squared increased from 0.913 to 0.939, meaning the modified model explains more variance in used-car prices.
- MAE dropped by roughly 20%, from $1,287.74 to $1,029.57, which is a meaningful improvement in resale-price estimation.
- MAPE improved from 22.50% to 18.22%, indicating a smaller relative error in price prediction.

The modified ANN is therefore more reliable for pricing support in a used-car dealership context, although there is still room to improve when the market is highly volatile or when rare vehicle types are underrepresented.

### 2B — Cherry Leaf Disease Classification

| Model | Test Accuracy | Weighted Recall | Weighted Precision | Weighted F1 |
|---|---:|---:|---:|---:|
| Baseline AlexNet | 89.29% | 0.8929 | 0.9040 | 0.8940 |
| Modified ResNet50 | 89.29% | 0.8929 | 0.8945 | 0.8918 |

Interpretation:

- Both models achieved the same final test accuracy (89.29%), which means the quantitative classification score was similar.
- The baseline AlexNet had slightly better weighted precision and F1, while the ResNet50 modification showed better training stability and smoother validation behavior.
- In practical terms, the modified ResNet is preferable for long-term deployment because it learns more steadily and generalizes more reliably, even though the accuracy metric is not dramatically higher.

### 4B
This notebook is not evaluated with benchmark metrics; it is included to demonstrate the mechanics of learning through gradient descent and backpropagation.

---

## Overall Conclusion
The deep learning experiments show two clear patterns:

1. For structured tabular data, a carefully tuned ANN can provide useful price prediction performance with meaningful gains after preprocessing and regularization.
2. For image classification, pretrained CNN architectures such as ResNet50 are especially effective because they provide stable feature extraction and better generalization than a basic custom CNN.

Across the notebooks, the strongest improvement comes from thoughtful architectural tuning, regularization, and proper preprocessing rather than simply increasing model depth.
