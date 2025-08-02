# College Student Placement Prediction

## Overview

This project focuses on predicting college student placement outcomes using machine learning techniques. The goal is to analyze various student attributes and determine whether a student is likely to be placed in a job or not.

## Dataset

The project uses a comprehensive dataset containing information about 10,000 college students with the following features:

### Features
- **College_ID**: Unique identifier for each college
- **IQ**: Intelligence Quotient score
- **Prev_Sem_Result**: Previous semester academic result
- **CGPA**: Cumulative Grade Point Average
- **Academic_Performance**: Academic performance rating (1-10)
- **Internship_Experience**: Whether the student has internship experience (Yes/No)
- **Extra_Curricular_Score**: Score in extracurricular activities (0-10)
- **Communication_Skills**: Communication skills rating (1-10)
- **Projects_Completed**: Number of projects completed
- **Placement**: Target variable - whether the student got placed (Yes/No)

## Data Preprocessing

1. **Data Loading**: Loaded the dataset using pandas
2. **Feature Engineering**: 
   - Removed the `College_ID` column as it's not relevant for prediction
   - Encoded categorical variables (`Placement` and `Internship_Experience`) to binary values (Yes=1, No=0)
3. **Data Balancing**: Applied Random Under Sampling to handle class imbalance
4. **Train-Test Split**: Split the data into 80% training and 20% testing sets

## Machine Learning Models

### 1. Logistic Regression
- **Accuracy**: ~84% on test set
- **Precision**: 0.842
- **Recall**: 0.862
- **F1 Score**: 0.852
- **AUC Score**: 0.930

### 2. Decision Tree Classifier
- **Accuracy**: 100% on both training and test sets
- **Precision**: 1.0
- **Recall**: 1.0
- **F1 Score**: 1.0
- **AUC Score**: 1.0

## Key Findings

1. **Data Quality**: The dataset contains 10,000 records with no missing values
2. **Feature Importance**: All features contribute to the prediction model
3. **Model Performance**: Decision Tree shows perfect performance, indicating potential overfitting
4. **Correlation Analysis**: Heatmap visualization shows relationships between features

## Visualizations

The project includes several visualizations:
- Correlation matrix heatmap
- ROC curve for model performance
- Confusion matrix for both models
- Decision tree visualization

## Files Structure

```
college_std_placement/
├── College_std_placement.ipynb          # Main Jupyter notebook
├── college_student_placement_dataset.csv # Dataset
├── env/                                 # Python virtual environment
└── README.md                           # This file
```

## Technologies Used

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms
- **Imbalanced-learn**: Handling class imbalance

## Installation and Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
   ```
4. Run the Jupyter notebook:
   ```bash
   jupyter notebook College_std_placement.ipynb
   ```

## Usage

1. Open the Jupyter notebook `College_std_placement.ipynb`
2. Run all cells sequentially to reproduce the analysis
3. The notebook includes data preprocessing, model training, and evaluation

## Model Deployment

The trained models can be used for predicting placement outcomes for new students. Example usage:

```python
# For Decision Tree model
input_data = pd.DataFrame([[107, 6.61, 6.28, 8, 0, 8, 8, 4]], 
                         columns=['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance', 
                                 'Internship_Experience', 'Extra_Curricular_Score', 
                                 'Communication_Skills', 'Projects_Completed'])
prediction = dt.predict(input_data)
```

## Future Improvements

1. **Feature Engineering**: Create additional features like GPA trends, skill combinations
2. **Model Selection**: Try other algorithms like Random Forest, SVM, or Neural Networks
3. **Hyperparameter Tuning**: Optimize model parameters using GridSearchCV
4. **Cross-validation**: Implement k-fold cross-validation for better model evaluation
5. **Feature Selection**: Analyze feature importance and remove less relevant features

## Contributing

Feel free to contribute to this project by:
- Improving the models
- Adding new features
- Enhancing visualizations
- Optimizing the code

## License

This project is open source and available under the MIT License.

---

**Note**: The Decision Tree model shows perfect performance, which might indicate overfitting. Consider using techniques like pruning or ensemble methods for better generalization. 