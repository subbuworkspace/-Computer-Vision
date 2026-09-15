# 👁️ Computer Vision

A complete beginner-to-interview guide to **Computer Vision (CV)** using Python, OpenCV, NumPy, Machine Learning, and Deep Learning.

---

## 📌 Table of Contents

1. What is Computer Vision?
2. Computer Vision vs Image Processing
3. How an Image is Represented
4. Pixels
5. RGB and Grayscale
6. Image Channels
7. Computer Vision Pipeline
8. Image Acquisition
9. Image Preprocessing
10. Image Resizing
11. Image Cropping
12. Image Rotation
13. Image Flipping
14. Color Space Conversion
15. Image Blurring
16. Thresholding
17. Edge Detection
18. Morphological Operations
19. Contours
20. Histogram
21. Feature Extraction
22. HOG
23. Object Detection
24. Image Classification
25. Image Segmentation
26. Face Detection
27. CNN
28. CNN Architecture
29. Pooling
30. Data Augmentation
31. Transfer Learning
32. Popular Computer Vision Models
33. OpenCV
34. scikit-image
35. Computer Vision with Machine Learning
36. Computer Vision with Deep Learning
37. Evaluation Metrics
38. Applications
39. Banking Use Cases
40. Advantages
41. Limitations
42. Mini Projects
43. Interview Questions
44. Quick Revision

---

# 1. What is Computer Vision?

**Computer Vision** is a field of Artificial Intelligence that enables computers to understand and process images and videos.

In simple words:

> **Computer Vision = Teaching a computer how to understand visual information.**

For example:

```text
Image
  ↓
Computer Vision
  ↓
Identify objects
  ↓
Understand the image
  ↓
Make a decision
```

### Example

A human sees:

```text
🐱
```

and understands:

> This is a cat.

A Computer Vision system receives pixels and uses image-processing or ML/DL techniques to predict:

```text
Class = Cat
Probability = 97%
```

---

# 2. Computer Vision vs Image Processing

These concepts are related but not exactly the same.

| Image Processing   | Computer Vision    |
| ------------------ | ------------------ |
| Manipulates images | Understands images |
| Enhancement        | Recognition        |
| Noise removal      | Object detection   |
| Resize             | Classification     |
| Blur               | Face recognition   |
| Thresholding       | Autonomous driving |

### Simple example

```text
Image Processing:

Dark Image
    ↓
Brightness Adjustment
    ↓
Better Image
```

Computer Vision:

```text
Image
  ↓
Model
  ↓
Dog
```

---

# 3. How an Image is Represented

A computer does not see an image like a human.

An image is represented as a collection of **numbers**.

For example:

```text
Image
 ↓
Pixels
 ↓
Numerical Matrix
```

A grayscale image can be represented as:

```text
[
 [0, 50, 100],
 [150, 200, 255],
 [100, 80, 20]
]
```

Where:

```text
0   → Black
255 → White
```

---

# 4. What is a Pixel?

A **pixel** is the smallest unit of a digital image.

For an 8-bit grayscale image:

```text
0   = Black
255 = White
```

Example:

```text
100 × 100 image

Number of pixels =
100 × 100
= 10,000 pixels
```

---

# 5. RGB Image

A color image usually contains three channels:

```text
R → Red
G → Green
B → Blue
```

Therefore an RGB image can be represented as:

```text
Height × Width × 3
```

Example:

```text
224 × 224 × 3
```

The `3` represents:

```text
Red
Green
Blue
```

---

# 6. Grayscale Image

A grayscale image has only one channel.

```text
Height × Width × 1
```

Example:

```text
224 × 224 × 1
```

Instead of three values per pixel, we have one intensity value.

---

# 7. Computer Vision Pipeline

A typical Computer Vision project looks like:

```text
Image / Video
      ↓
Data Collection
      ↓
Image Preprocessing
      ↓
Feature Extraction
      ↓
Model
      ↓
Prediction
      ↓
Evaluation
```

For Deep Learning:

```text
Image
  ↓
Resize
  ↓
Normalize
  ↓
Data Augmentation
  ↓
CNN
  ↓
Classification / Detection / Segmentation
```

---

# 8. Image Acquisition

First we need an image.

Sources can include:

* Camera
* Mobile
* CCTV
* Satellite
* Medical scanner
* Documents
* Internet datasets
* Video

Python libraries commonly used:

```text
OpenCV
Pillow
scikit-image
TensorFlow
PyTorch
TorchVision
```

---

# 9. OpenCV

**OpenCV** stands for:

> Open Source Computer Vision Library

It is one of the most commonly used libraries for computer vision and image processing.

Install:

```bash
pip install opencv-python
```

Import:

```python
import cv2
```

---

# 10. Read an Image

```python
import cv2

img = cv2.imread("image.jpg")

print(img.shape)
```

Example output:

```text
(500, 700, 3)
```

Meaning:

```text
Height = 500
Width  = 700
Channels = 3
```

---

# 11. Display an Image

```python
import cv2

img = cv2.imread("image.jpg")

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

# 12. Image Resizing

Images often have different dimensions.

We can resize them:

```python
img = cv2.imread("image.jpg")

resize = cv2.resize(img, (224, 224))

cv2.imshow("Image", resize)
cv2.waitKey(0)
```

Why resize?

Because ML/DL models usually require a fixed input size.

Example:

```text
Original
1920 × 1080

        ↓

Resize

224 × 224
```

---

# 13. Image Cropping

Cropping means selecting a particular region.

```python
img = cv2.imread("image.jpg")

crop = img[100:300, 200:400]

cv2.imshow("Crop", crop)
cv2.waitKey(0)
```

---

# 14. Image Rotation

```python
img = cv2.imread("image.jpg")

rotate = cv2.rotate(
    img,
    cv2.ROTATE_90_CLOCKWISE
)

cv2.imshow("Rotated", rotate)
cv2.waitKey(0)
```

---

# 15. Image Flipping

```python
flip = cv2.flip(img, 1)
```

Common values:

```text
0  → Vertical
1  → Horizontal
-1 → Both
```

---

# 16. RGB vs BGR

One important interview question:

### What color order does OpenCV use?

OpenCV normally uses:

```text
BGR
```

not:

```text
RGB
```

Example:

```python
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

---

# 17. Grayscale Conversion

```python
gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)
```

Why convert to grayscale?

Because many operations such as:

* Edge detection
* Thresholding
* Contour detection

do not require full color information.

---

# 18. Image Normalization

Pixel values are commonly:

```text
0 → 255
```

For neural networks, we often scale them:

```python
img = img / 255.0
```

Now:

```text
0   → 0.0
255 → 1.0
```

This can make neural-network training easier.

---

# 19. Image Blurring

Images can contain noise.

Blurring can help reduce noise before operations such as edge detection.

### Gaussian Blur

```python
blur = cv2.GaussianBlur(
    img,
    (5, 5),
    0
)
```

Other filters include:

```text
Gaussian Filter
Median Filter
Bilateral Filter
```

---

# 20. Thresholding

Thresholding converts an image into a simpler representation, often binary.

Example:

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)
```

Concept:

```text
Pixel < 127 → 0
Pixel > 127 → 255
```

OpenCV also provides:

* Simple thresholding
* Adaptive thresholding
* Otsu thresholding

---

# 21. Edge Detection

An edge represents a strong change in intensity.

Example:

```text
Object
   ↓
Boundary
   ↓
Edge
```

Common algorithms:

```text
Sobel
Laplacian
Canny
```

### Canny Example

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(
    gray,
    100,
    200
)
```

Canny is commonly used to identify object boundaries.

---

# 22. Morphological Operations

Morphological operations work mainly with shapes in binary images.

Important operations:

```text
Erosion
Dilation
Opening
Closing
```

### Erosion

Removes small white regions.

### Dilation

Expands white regions.

### Opening

```text
Erosion → Dilation
```

Useful for removing noise.

### Closing

```text
Dilation → Erosion
```

Useful for filling small gaps.

---

# 23. Contours

A contour is a curve representing the boundary of an object.

Example:

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(
    gray, 127, 255,
    cv2.THRESH_BINARY
)

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)
```

Contours are useful for:

* Shape analysis
* Object detection
* Object recognition
* Finding boundaries

OpenCV recommends using a suitable binary image before contour extraction.

---

# 24. Histogram

An image histogram shows the distribution of pixel intensity values.

For grayscale:

```text
X-axis → Pixel intensity
Y-axis → Number of pixels
```

Example:

```python
import cv2
import matplotlib.pyplot as plt

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.hist(gray.ravel(), 256, [0, 256])
plt.show()
```

Histogram techniques are useful for:

* Contrast analysis
* Image enhancement
* Histogram equalization

---

# 25. Feature Extraction

Feature extraction means finding useful information from an image.

Examples:

```text
Edges
Corners
Texture
Shape
Color
Patterns
```

Traditional Computer Vision:

```text
Image
 ↓
Feature Extraction
 ↓
Machine Learning
 ↓
Prediction
```

Deep Learning:

```text
Image
 ↓
CNN
 ↓
Automatically learns features
 ↓
Prediction
```

---

# 26. HOG

**HOG = Histogram of Oriented Gradients**

HOG describes the shape of objects using gradient directions.

It was widely used in traditional object detection, especially pedestrian detection.

Concept:

```text
Image
 ↓
Gradient
 ↓
Orientation
 ↓
Histogram
 ↓
Feature Vector
```

HOG is useful for understanding how traditional Computer Vision extracted features before deep CNNs became dominant.

---

# 27. Image Classification

Image classification answers:

> What is in this image?

Example:

```text
Image
 ↓
Model
 ↓
Cat
```

or:

```text
Image
 ↓
Model
 ↓
Dog
```

### Multi-class example

```text
Cat
Dog
Horse
Bird
```

The model predicts one or more classes depending on the problem design.

---

# 28. Object Detection

Object detection answers two questions:

1. **What object is present?**
2. **Where is the object?**

Example:

```text
Image
 ↓
Object Detection
 ↓
Dog → Bounding Box
Car → Bounding Box
Person → Bounding Box
```

Output usually contains:

```text
Class
Bounding Box
Confidence Score
```

Popular approaches include:

```text
YOLO
Faster R-CNN
SSD
RetinaNet
FCOS
```

TorchVision currently provides pretrained models for object detection and other vision tasks.

---

# 29. Image Classification vs Object Detection

| Classification    | Object Detection          |
| ----------------- | ------------------------- |
| What is present?  | What + where?             |
| Usually one label | Multiple objects possible |
| No bounding box   | Bounding boxes            |
| Cat               | Cat + location            |
| Easier problem    | More complex              |

Example:

```text
Classification:

[ Dog ]

Detection:

[ Dog ] → Bounding Box
[ Person ] → Bounding Box
[ Car ] → Bounding Box
```

---

# 30. Image Segmentation

Segmentation assigns pixels to regions or objects.

There are two important types:

### Semantic Segmentation

Every pixel gets a class.

```text
Road → Road
Car → Car
Person → Person
```

Objects of the same class are generally treated as the same category.

### Instance Segmentation

Different objects are separated.

```text
Person 1 → Mask 1
Person 2 → Mask 2
Person 3 → Mask 3
```

---

# 31. Classification vs Detection vs Segmentation

| Task                  | Output               |
| --------------------- | -------------------- |
| Classification        | Class                |
| Detection             | Class + Bounding Box |
| Semantic Segmentation | Class per pixel      |
| Instance Segmentation | Object-specific mask |

Remember:

```text
Classification → WHAT
Detection     → WHAT + WHERE
Segmentation  → EXACT PIXELS
```

---

# 32. Face Detection

Face detection identifies faces in an image or video.

Traditional OpenCV example:

```python
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

faces = face_cascade.detectMultiScale(
    gray,
    1.1,
    4
)
```

Then draw rectangles:

```python
for x, y, w, h in faces:
    cv2.rectangle(
        img,
        (x, y),
        (x+w, y+h),
        (255, 0, 0),
        2
    )
```

---

# 33. CNN

**CNN = Convolutional Neural Network**

CNNs are one of the most important Deep Learning architectures for Computer Vision.

A CNN automatically learns useful visual features from images.

Basic architecture:

```text
Input Image
     ↓
Convolution
     ↓
Activation
     ↓
Pooling
     ↓
Convolution
     ↓
Pooling
     ↓
Flatten
     ↓
Fully Connected Layer
     ↓
Output
```

TensorFlow provides CNN examples for image classification such as CIFAR-10.

---

# 34. Convolution

Convolution applies a small matrix called a **filter/kernel** to an image.

Example:

```text
Image
 ↓
Kernel
 ↓
Feature Map
```

Different kernels can learn:

```text
Edges
Corners
Textures
Patterns
Shapes
```

In deep CNNs:

```text
Early layers
 ↓
Edges

Middle layers
 ↓
Textures / shapes

Deep layers
 ↓
Objects / high-level patterns
```

---

# 35. Pooling

Pooling reduces the spatial dimensions of feature maps.

Common types:

```text
Max Pooling
Average Pooling
```

### Max Pooling

Selects the maximum value from a region.

Example:

```text
1  5
3  2

↓

5
```

Advantages:

* Reduces computation
* Reduces dimensions
* Helps retain important features

---

# 36. Flattening

CNN feature maps need to be converted into a vector before entering traditional fully connected layers.

```text
Feature Maps
     ↓
Flatten
     ↓
1D Vector
     ↓
Dense Layer
```

---

# 37. Simple CNN Example

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu",
                  input_shape=(64,64,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(2, activation="softmax")
])

model.summary()
```

---

# 38. Data Augmentation

Data augmentation creates variations of existing training images.

Examples:

```text
Rotation
Flipping
Zoom
Translation
Cropping
Brightness changes
```

Example:

```python
from tensorflow.keras import layers

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])
```

Why?

```text
More image variation
       ↓
Better generalization
       ↓
Less overfitting
```

TensorFlow's image tutorials specifically use augmentation as one technique for improving generalization and reducing overfitting.

---

# 39. Transfer Learning

Transfer Learning means using knowledge learned from an existing pretrained model for a new task.

Example:

```text
Pretrained Model
      ↓
ImageNet knowledge
      ↓
New Dataset
      ↓
Fine-tune
      ↓
New Computer Vision Task
```

Popular pretrained models:

```text
ResNet
VGG
MobileNet
EfficientNet
DenseNet
Inception
ConvNeXt
Vision Transformer
```

For a small dataset, transfer learning is often much more practical than training a large CNN from scratch. TensorFlow demonstrates feature extraction and fine-tuning with pretrained MobileNetV2.

---

# 40. Feature Extraction vs Fine-Tuning

### Feature Extraction

Freeze the pretrained network.

```text
Pretrained Model
      ↓
Frozen Layers
      ↓
New Classifier
```

### Fine-Tuning

Unfreeze some upper layers and train them with the new task.

```text
Pretrained Model
      ↓
Freeze lower layers
      ↓
Train upper layers
      ↓
New Classifier
```

---

# 41. Popular Computer Vision Models

### Image Classification

```text
LeNet
AlexNet
VGG
ResNet
DenseNet
Inception
EfficientNet
MobileNet
ConvNeXt
Vision Transformer
```

### Object Detection

```text
YOLO
R-CNN
Fast R-CNN
Faster R-CNN
SSD
RetinaNet
FCOS
```

### Segmentation

```text
U-Net
Mask R-CNN
DeepLab
```

TorchVision currently provides pretrained architectures for classification, semantic segmentation, object detection, instance segmentation, keypoint detection, video classification, and optical flow.

---

# 42. OpenCV vs TensorFlow/PyTorch

| OpenCV           | TensorFlow / PyTorch |
| ---------------- | -------------------- |
| Image processing | Deep Learning        |
| Resize           | CNN                  |
| Blur             | Classification       |
| Threshold        | Object Detection     |
| Edge Detection   | Segmentation         |
| Contours         | Transfer Learning    |
| Video processing | Neural Networks      |

In a real project they can be used together.

```text
OpenCV
  ↓
Preprocessing
  ↓
CNN / Deep Learning
  ↓
Prediction
```

---

# 43. scikit-image

`scikit-image` is another Python library for image processing and computer vision. It works with NumPy arrays and provides functionality for filtering, segmentation, morphology, feature detection, geometric transformations and more.

Install:

```bash
pip install scikit-image
```

Example:

```python
from skimage import data

image = data.camera()

print(image.shape)
```

---

# 44. Computer Vision with Machine Learning

Before Deep Learning became dominant, a common workflow was:

```text
Image
 ↓
Preprocessing
 ↓
Feature Extraction
 ↓
Feature Vector
 ↓
Machine Learning
 ↓
Prediction
```

Example:

```text
Image
 ↓
HOG
 ↓
Feature Vector
 ↓
SVM
 ↓
Person / Not Person
```

Algorithms commonly used:

```text
SVM
Random Forest
KNN
Logistic Regression
Decision Tree
```

---

# 45. Computer Vision with Deep Learning

Modern workflow:

```text
Image
 ↓
Preprocessing
 ↓
CNN
 ↓
Automatic Feature Extraction
 ↓
Classification
```

The major difference:

### Traditional ML

```text
Human designs features
       ↓
ML model
```

### Deep Learning

```text
CNN learns features
       ↓
Prediction
```

---

# 46. Evaluation Metrics

For image classification:

### Accuracy

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

```text
Precision =
TP / (TP + FP)
```

### Recall

```text
Recall =
TP / (TP + FN)
```

### F1 Score

```text
F1 =
2 × Precision × Recall
-----------------------
Precision + Recall
```

For object detection, commonly used metrics include:

```text
IoU
Precision
Recall
mAP
```

---

# 47. IoU

**IoU = Intersection over Union**

Used heavily in object detection and segmentation.

```text
IoU =
Area of Intersection
--------------------
Area of Union
```

Conceptually:

```text
Predicted Box
      ∩
Actual Box
      ↓
Intersection

Predicted Box
      ∪
Actual Box
      ↓
Union
```

Higher IoU means better overlap.

---

# 48. Computer Vision Project Pipeline

A practical project can follow:

```text
1. Collect Images
       ↓
2. Label Data
       ↓
3. Clean Data
       ↓
4. Resize Images
       ↓
5. Normalize
       ↓
6. Augmentation
       ↓
7. Train / Validation / Test Split
       ↓
8. Train Model
       ↓
9. Evaluate
       ↓
10. Improve
       ↓
11. Deploy
```

---

# 49. Common Computer Vision Problems

### Problem 1: Classification

```text
Is this image a cat or dog?
```

### Problem 2: Detection

```text
Where are all the cars?
```

### Problem 3: Segmentation

```text
Which pixels belong to the road?
```

### Problem 4: Face Detection

```text
Where is the face?
```

### Problem 5: OCR

```text
What text is present in the image?
```

### Problem 6: Object Tracking

```text
Where does the same object move in the video?
```

---

# 50. OCR

**OCR = Optical Character Recognition**

OCR converts text from an image into machine-readable text.

Example:

```text
Image
 ↓
Preprocessing
 ↓
Text Detection
 ↓
OCR
 ↓
"Invoice Amount: ₹5000"
```

Popular OCR tools:

```text
Tesseract
EasyOCR
PaddleOCR
Cloud Vision APIs
```

A typical OCR pipeline:

```text
Document
 ↓
Grayscale
 ↓
Noise Removal
 ↓
Thresholding
 ↓
Text Detection
 ↓
OCR
 ↓
Text
```

---

# 51. Computer Vision + Banking

Computer Vision has many possible banking applications.

### 1. KYC

```text
ID Document
     ↓
Computer Vision
     ↓
Document Detection
     ↓
OCR
     ↓
Customer Information
```

### 2. Cheque Processing

```text
Cheque Image
     ↓
Image Processing
     ↓
OCR
     ↓
Amount / Account Information
```

### 3. Signature Verification

```text
Signature Image
       ↓
Feature Extraction
       ↓
Similarity Model
       ↓
Match / No Match
```

### 4. Document Classification

```text
Document
 ↓
Computer Vision
 ↓
Invoice / Passport / Bank Statement / Other
```

### 5. Fraud Detection

Computer Vision can support:

```text
Document verification
Identity verification
Altered document detection
Signature analysis
```

---

# 52. Common Computer Vision Applications

```text
Face Recognition
Object Detection
Self-driving Cars
Medical Imaging
OCR
Surveillance
Satellite Image Analysis
Industrial Inspection
Retail Analytics
Robotics
Gesture Recognition
Augmented Reality
Autonomous Systems
Document Processing
```

---

# 53. Advantages

* Automates visual inspection
* Can process large numbers of images
* Useful for real-time applications
* Can detect objects automatically
* Supports document automation
* Works with images and videos
* Deep Learning can learn complex visual features

---

# 54. Limitations

Computer Vision can struggle with:

```text
Poor lighting
Blurred images
Occlusion
Low-quality images
Unusual viewpoints
Small datasets
Class imbalance
Domain shift
Adversarial examples
```

Deep Learning models can also require significant:

```text
Data
Computational resources
Training time
```

---

# 55. Small OpenCV Project

## Edge Detection

```python
import cv2

img = cv2.imread("image.jpg")

gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)

edges = cv2.Canny(
    gray,
    100,
    200
)

cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Pipeline

```text
Image
 ↓
Grayscale
 ↓
Canny
 ↓
Edges
```

---

# 56. Small CNN Project

Example project:

## Cat vs Dog Classification

```text
Dataset
   ↓
Resize images
   ↓
Normalize
   ↓
Data Augmentation
   ↓
CNN
   ↓
Train
   ↓
Evaluate
   ↓
Cat / Dog
```

Possible folder structure:

```text
Computer-Vision/
│
├── README.md
├── dataset/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── models/
└── requirements.txt
```

---

# 57. Computer Vision Project Ideas

### Beginner

1. Image Resizer
2. Image Grayscale Converter
3. Edge Detection
4. Image Blur Application
5. Face Detection
6. Color Detection
7. Shape Detection
8. Document Scanner

### Intermediate

9. OCR Application
10. Number Plate Detection
11. Object Detection
12. Face Recognition
13. Hand Gesture Recognition
14. Image Classification
15. People Counter

### Advanced

16. YOLO Object Detection
17. Semantic Segmentation
18. Instance Segmentation
19. Face Recognition System
20. Real-Time Object Tracking
21. Medical Image Classification
22. Autonomous Driving Vision
23. Document Fraud Detection

---

# 58. Important Python Libraries

```text
OpenCV
NumPy
Pillow
Matplotlib
scikit-image
TensorFlow
Keras
PyTorch
TorchVision
```

Example imports:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

For Deep Learning:

```python
import tensorflow as tf
```

or:

```python
import torch
import torchvision
```

---

# 59. Important Interview Questions

## Basic Questions

### 1. What is Computer Vision?

Computer Vision is an AI field that enables computers to process and understand visual information such as images and videos.

### 2. What is an image?

An image is a matrix/array of pixel values.

### 3. What is a pixel?

A pixel is the smallest unit of a digital image.

### 4. What is RGB?

RGB represents:

```text
Red
Green
Blue
```

### 5. What is grayscale?

A grayscale image contains intensity information using a single channel.

### 6. What is OpenCV?

OpenCV is an open-source computer vision and image-processing library.

### 7. What is image preprocessing?

Preparing images before sending them to an algorithm/model.

Examples:

```text
Resize
Normalize
Crop
Blur
Grayscale
Threshold
```

---

# 60. Image Processing Interview Questions

### 8. Why convert an image to grayscale?

To reduce information and computation when color is not required.

### 9. What is thresholding?

Thresholding converts pixels based on a threshold value, often producing a binary image.

### 10. What is edge detection?

Finding strong intensity changes that often correspond to object boundaries.

### 11. What is Canny Edge Detection?

A multi-stage edge detection algorithm commonly used to identify edges.

### 12. What are contours?

Contours represent continuous boundaries of objects.

### 13. What is image segmentation?

Dividing an image into meaningful regions.

### 14. What is image augmentation?

Creating variations of training images to improve generalization.

### 15. What is normalization?

Scaling pixel values to a smaller numerical range, commonly 0–1.

---

# 61. CNN Interview Questions

### 16. What is CNN?

Convolutional Neural Network.

It is a neural network architecture widely used for image-related tasks.

### 17. Why CNN instead of a normal neural network?

CNNs exploit spatial structure and can learn local visual patterns efficiently.

### 18. What is convolution?

Applying a learnable filter/kernel across an image or feature map.

### 19. What is a kernel?

A small matrix used to detect patterns in an image.

### 20. What is a feature map?

The output produced after applying convolution filters.

### 21. What is pooling?

A technique that reduces spatial dimensions of feature maps.

### 22. Max pooling vs average pooling?

```text
Max Pooling
→ Maximum value

Average Pooling
→ Average value
```

### 23. Why use ReLU?

ReLU introduces non-linearity and is commonly used in neural networks.

```text
ReLU(x) = max(0, x)
```

### 24. What is flattening?

Converting multidimensional feature maps into a one-dimensional vector.

---

# 62. Object Detection Interview Questions

### 25. What is object detection?

Identifying objects and their locations in an image.

### 26. What is a bounding box?

A rectangle surrounding a detected object.

### 27. What is confidence score?

The model's estimated confidence in a prediction.

### 28. What is IoU?

Intersection over Union measures overlap between predicted and ground-truth regions.

### 29. What is NMS?

**NMS = Non-Maximum Suppression**

It removes redundant overlapping bounding boxes.

### 30. Classification vs object detection?

```text
Classification
→ What?

Detection
→ What + Where?
```

---

# 63. Deep Learning Interview Questions

### 31. What is transfer learning?

Using knowledge from a pretrained model for a new task.

### 32. What is fine-tuning?

Continuing training of selected pretrained layers on a new dataset.

### 33. Why use transfer learning?

Because training a large vision model from scratch can require substantial data and computation.

### 34. What is overfitting in Computer Vision?

When a model performs well on training images but poorly on unseen images.

### 35. How can you reduce overfitting?

```text
Data Augmentation
Dropout
Regularization
Early Stopping
Transfer Learning
More Data
```

### 36. What is data augmentation?

Generating realistic variations of training images.

---

# 64. Advanced Interview Questions

### 37. Why are CNNs good for images?

Because convolution captures local spatial patterns and builds hierarchical representations.

### 38. Why do deeper CNN layers learn different features?

Earlier layers tend to learn simple patterns, while deeper layers combine them into more complex representations.

### 39. What happens if image size is too large?

Training and inference can become computationally expensive.

### 40. What happens if image size is too small?

Important visual information may be lost.

### 41. Why normalize image pixels?

It can improve numerical stability and make neural-network optimization easier.

### 42. What is class imbalance?

When some image classes contain many more examples than others.

Example:

```text
Cats = 9,000
Dogs = 1,000
```

### 43. How can you handle class imbalance?

```text
Class weights
Oversampling
Augmentation
Undersampling
Better evaluation metrics
```

### 44. What is semantic segmentation?

Assigning a class to each pixel.

### 45. What is instance segmentation?

Separating individual object instances at pixel level.

---

# 65. Scenario-Based Interview Questions

### 46. You have only 1,000 images. Would you train CNN from scratch?

Usually, I would first consider **transfer learning** using a pretrained model and then fine-tune it if appropriate.

### 47. Your training accuracy is 98% but validation accuracy is 70%. What is happening?

Likely overfitting.

Possible solutions:

```text
Data Augmentation
Dropout
Regularization
Early Stopping
Transfer Learning
More Training Data
```

### 48. Your images have different sizes. What will you do?

Resize them to a consistent model input size.

### 49. Your images are very dark. What can you do?

Possible techniques:

```text
Brightness adjustment
Histogram equalization
CLAHE
Image enhancement
```

### 50. You need to identify every car in an image. Which task?

**Object Detection.**

### 51. You need to classify an image as Cat or Dog. Which task?

**Image Classification.**

### 52. You need the exact pixels belonging to a car. Which task?

**Image Segmentation.**

---

# 66. One-Minute Interview Answer

> "Computer Vision is a field of AI that enables machines to understand images and videos. A typical computer vision pipeline starts with image acquisition, followed by preprocessing such as resizing, grayscale conversion, normalization and augmentation. Traditional computer vision can use techniques such as edge detection, thresholding, contours and feature extraction, followed by machine learning algorithms such as SVM or Random Forest. Modern computer vision commonly uses deep learning, especially CNNs, for image classification, object detection and segmentation. OpenCV is widely used for image processing, while TensorFlow and PyTorch are commonly used for deep learning-based computer vision."

---

# 67. Quick Revision

Remember this:

```text
COMPUTER VISION
       │
       ├── Image Processing
       │     ├── Resize
       │     ├── Crop
       │     ├── Blur
       │     ├── Threshold
       │     ├── Edge Detection
       │     └── Contours
       │
       ├── Traditional ML
       │     ├── HOG
       │     ├── SVM
       │     ├── Random Forest
       │     └── KNN
       │
       └── Deep Learning
             ├── CNN
             ├── Transfer Learning
             ├── Classification
             ├── Object Detection
             └── Segmentation
```

---

# 68. Most Important Concepts to Remember

```text
Pixel
Image Matrix
RGB
Grayscale
BGR
Resize
Normalization
Thresholding
Blurring
Edge Detection
Canny
Contours
Morphology
Histogram
HOG
CNN
Convolution
Kernel
Feature Map
Pooling
Flatten
Data Augmentation
Transfer Learning
Classification
Object Detection
Bounding Box
IoU
Segmentation
OCR
Object Tracking
```

---

# 69. Final Cheat Sheet

| Concept           | Remember                          |
| ----------------- | --------------------------------- |
| Pixel             | Smallest image unit               |
| RGB               | Red + Green + Blue                |
| Grayscale         | One intensity channel             |
| OpenCV            | Image processing / CV             |
| Threshold         | Pixel classification by intensity |
| Canny             | Edge detection                    |
| Contour           | Object boundary                   |
| HOG               | Traditional feature descriptor    |
| CNN               | Deep learning for visual patterns |
| Convolution       | Extracts local features           |
| Pooling           | Reduces spatial size              |
| Classification    | What?                             |
| Detection         | What + Where?                     |
| Segmentation      | Which pixels?                     |
| IoU               | Bounding-region overlap           |
| Augmentation      | Create training variations        |
| Transfer Learning | Reuse pretrained knowledge        |
| Fine-Tuning       | Adapt pretrained layers           |

---

# 🚀 Recommended Learning Order

If you are learning Computer Vision from beginner level, follow this order:

```text
1. Python
      ↓
2. NumPy
      ↓
3. Image Basics
      ↓
4. OpenCV
      ↓
5. Image Processing
      ↓
6. Thresholding
      ↓
7. Edge Detection
      ↓
8. Contours
      ↓
9. HOG + Traditional ML
      ↓
10. CNN
      ↓
11. Data Augmentation
      ↓
12. Transfer Learning
      ↓
13. Image Classification
      ↓
14. Object Detection
      ↓
15. Segmentation
      ↓
16. OCR
      ↓
17. Real-Time Computer Vision
```

---

# 🎯 Best Beginner Portfolio Projects

Start with these:

### Project 1

**OpenCV Image Processing**

```text
Read Image
→ Resize
→ Grayscale
→ Blur
→ Canny
→ Display
```

### Project 2

**Face Detection**

```text
Camera
→ Face Detection
→ Bounding Box
```

### Project 3

**Document Scanner**

```text
Document
→ Edge Detection
→ Contours
→ Perspective Transform
→ Scanned Document
```

### Project 4

**Cat vs Dog CNN**

```text
Images
→ Preprocessing
→ CNN
→ Classification
```

### Project 5

**Object Detection**

```text
Image
→ YOLO / Faster R-CNN
→ Objects + Bounding Boxes
```

### Project 6

**OCR Document Reader**

```text
Document
→ Preprocessing
→ OCR
→ Extracted Text
```

---

# ⭐ Final Interview Memory Trick

```text
Image Processing
        ↓
Prepare the image

Computer Vision
        ↓
Understand the image

Machine Learning
        ↓
Use extracted features

Deep Learning
        ↓
Learn features automatically
```

And remember:

```text
CLASSIFICATION
      ↓
     WHAT?

DETECTION
      ↓
WHAT + WHERE?

SEGMENTATION
      ↓
WHICH PIXELS?

OCR
      ↓
WHAT TEXT?

TRACKING
      ↓
WHERE IS THE OBJECT MOVING?
```

---

## 📚 Useful Libraries

```text
OpenCV
NumPy
Pillow
Matplotlib
scikit-image
TensorFlow / Keras
PyTorch
TorchVision
```

This README is designed as a foundation for a **Computer Vision GitHub repository** and can be expanded later with dedicated projects for OpenCV, CNN, YOLO, OCR, face detection, and image classification.
