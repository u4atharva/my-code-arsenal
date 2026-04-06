<div align="center">
  <h1>🚀 My Code Arsenal</h1>
  <p><i>A curated collection of specialized projects spanning Computer Vision, Machine Learning, and Custom Utilities.</i></p>

  [![Python](https://img.shields.io/badge/Python-3.x-blue.svg?logo=python&style=for-the-badge)](https://www.python.org/)
  [![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-red.svg?logo=opencv&style=for-the-badge)](https://opencv.org/)
  [![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg?logo=scikit-learn&style=for-the-badge)](https://scikit-learn.org/)
  [![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg?logo=pandas&style=for-the-badge)](https://pandas.pydata.org/)
</div>

---

## 📂 Project Structure

This repository is divided into three distinct modules, each demonstrating a different application of Python:

### 1. 👁️ Computer Vision: Rubik's Cube Color Thresholding
**Location:** `Computer Vision/SolveCube/`

An OpenCV-based interactive utility designed to help process and calibrate images of a Rubik's Cube.
- **What it does**: Provides a user-friendly GUI with trackbars to perform dynamic HSV color thresholding on cube faces.
- **Highlights**: 
  - Live masking preview of the HSV image.
  - Indispensable for color tuning and calibration before applying automated cube-solving algorithms (like Kociemba's algorithm).
- **Tech Stack**: `OpenCV` (`cv2`), `Python`.

### 2. 🧠 Machine Learning: Hotel Booking Cancellations
**Location:** `Machine Learning/HotelPrices/venv/`

A complete end-to-end data science pipeline for analyzing hotel booking data and predicting reservation cancellations.
- **What it does**: Cleans, preprocesses, and models a hotel bookings dataset to predict whether a customer will cancel their reservation (`is_canceled`).
- **Highlights**: 
  - **Data Preprocessing**: Handling missing values, date-time manipulation, and One-Hot Encoding for categorical variables.
  - **Classification Models**: Implements and evaluates a **Support Vector Machine (SVM)** and a **Random Forest Classifier**.
  - **Feature Insights**: Derives Feature Importance to identify the primary factors driving a guest's decision to cancel.
- **Tech Stack**: `Pandas`, `NumPy`, `Scikit-Learn` (`svm.SVC`, `RandomForestClassifier`), `Matplotlib`.

### 3. 🛠️ Custom Utilities: PDF Batch Unlocker
**Location:** `Custom Utilities/`

Handy, time-saving scripts tailored for automating mundane PDF management tasks.
- **What it does**: Includes tools (`UnlockPDF.py` & `removePdfPassword.py`) that batch-process directories containing password-protected PDF files, unlock them using a master password, and save them to an output directory.
- **Tech Stack**: `pikepdf`, `os`.

---

## ⚙️ Getting Started

### Prerequisites

Ensure you have Python 3.x installed. You can install all necessary dependencies for the entire repository using pip:

```bash
pip install opencv-python pandas numpy scikit-learn matplotlib pikepdf
```

### Running the Projects

**1. Computer Vision (Cube Color Calibration):**
Navigate into the directory and run the script:
```bash
cd "Computer Vision/SolveCube"
python ProcessFace.py
```

**2. Machine Learning (Hotel Prices Prediction):**
Navigate to the directory holding the ML script (make sure `hotel_bookings.csv` is present in the same folder):
```bash
cd "Machine Learning/HotelPrices/venv"
python program.py
```

**3. Custom Utilities (PDF Unlocker):**
Open the scripts and configure your `Input` / `Output` folder paths and the master password. Then execute:
```bash
cd "Custom Utilities"
python UnlockPDF.py
# or
python removePdfPassword.py
```

---

<p align="center">
  <i>Maintained to build a strong arsenal of reusable code snippets and modular algorithms.</i>
</p>