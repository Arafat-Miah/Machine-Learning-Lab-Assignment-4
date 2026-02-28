# Machine Learning Lab - Assignment 4: Classification

This repository contains my solutions for **Assignment 4** of the Machine Learning course (521289S). The focus of this assignment is **Classification**, covering fundamental concepts from linear decision boundaries to nonlinear feature transformations and handling imbalanced data.

## 📂 Repository Structure

The solutions are organized into separate MATLAB scripts corresponding to each task in the assignment:

| File Name | Description |
| :--- | :--- |
| `task1_Cost_Functions_For_Classification.m` | Implementation of Step & Sigmoid models and Least Squares vs. Cross-Entropy cost functions. |
| `task2_Perceptron_Breast_Cancer_Classification.m` | **Linear Classification**: Training a Perceptron using the **Regularized Softmax** cost function to ensure convergence on linearly separable data. |
| `task3_Perceptron_Spam_Detection.m` | **Data Normalization**: Spam detection where Z-score normalization is critical for Gradient Descent convergence. |
| `task4_Weighted_Classification.m` | **Class Imbalance**: Implementing a **Weighted Softmax** cost to handle datasets with unequal class distribution. |
| `task5_Nonlinear_Classification.m` | **Nonlinear Classification**: Using feature transformation ($x \to x^2$) to fit an elliptical decision boundary. |

---

## 📝 Task Details

### Task 1: Cost Functions for Classification
Explored the mathematical foundations of classification.
- Implemented the **Step function** (discontinuous) and **Sigmoid function** (smooth approximation).
- Compared **Least Squares (LS)** cost against **Cross-Entropy (CE)** cost.
- **Key Insight**: Cross-Entropy is generally preferred for classification because it penalizes confident wrong predictions more heavily and is convex for logistic regression.

### Task 2: Perceptron Breast Cancer Classification
Implemented a linear classifier for a medical dataset.
- **Challenge**: Standard Softmax can diverge (weights $\to \infty$) if classes are perfectly separable.
- **Solution**: Implemented **Regularized Softmax Cost** ($\lambda \|w\|^2$) to constrain the weights and ensure a stable solution.

### Task 3: Perceptron Spam Detection
Applied the classifier to a high-dimensional spam dataset.
- **Challenge**: Features had vastly different scales (e.g., word frequency vs. capital letter length), causing Gradient Descent to zig-zag.
- **Solution**: Implemented **Z-score Normalization** (`normalize(X)`) to scale all features to zero mean and unit variance before training.

### Task 4: Weighted Classification
Handled a dataset with a severe class imbalance (many samples of one class, few of the other).
- **Challenge**: A standard classifier tends to ignore the minority class to maximize overall accuracy.
- **Solution**: Implemented **Weighted Softmax**, where individual samples are assigned weights ($\beta_p$) to increase the penalty for misclassifying the minority class.

### Task 5: Nonlinear Classification
Solved a problem where the data was not linearly separable (concentric circles).
- **Challenge**: A linear line ($w^Tx$) cannot separate an inner circle from an outer ring.
- **Solution**: Used **Feature Transformation**. By mapping features from $x$ to $x^2$, the problem became linear in the transformed space, allowing the model to fit an elliptical boundary.

---

## ⚠️ Repository Purpose & Academic Integrity

This repository is created solely to demonstrate the knowledge and practical skills I gained in machine learning optimization during this course.

**The code is:**

* ❌ **Not intended for reuse, redistribution, or submission by others**
* ❌ **Not shared for the purpose of passing coursework or assessments**
* ✅ **Maintained as a personal academic and technical portfolio artifact**

Any use of this material should respect academic integrity policies and course regulations.
