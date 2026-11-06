
import pandas as pd
import matplotlib.pyplot as plt

# Entering the results from the four models
results = pd.DataFrame({
    'Model': [
        'Linear Regression',
        'KNN Regressor',
        'Decision Tree',
        'Random Forest'
    ],
    'MAE': [
        30.0605,
        23.7143,
        28.4881,
        20.4268
    ],
    'RMSE': [
        56.7973,
        47.1965,
        57.9591,
        40.3555
    ],
    'R2 Score': [
        0.8238,
        0.8784,
        0.8165,
        0.9111
    ]
})

# Displaying the model comparison table
print("\nModel Comparison Results:")
print(results.to_string(index=False))

# Comparing Mean Absolute Error
plt.figure(figsize=(8, 5))
plt.bar(results['Model'], results['MAE'])
plt.title('Model Comparison - MAE')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Comparing Root Mean Squared Error
plt.figure(figsize=(8, 5))
plt.bar(results['Model'], results['RMSE'])
plt.title('Model Comparison - RMSE')
plt.ylabel('Root Mean Squared Error')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Comparing R2 Score
plt.figure(figsize=(8, 5))
plt.bar(results['Model'], results['R2 Score'])
plt.title('Model Comparison - R2 Score')
plt.ylabel('R2 Score')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Finding the best model
best_model = results.loc[results['R2 Score'].idxmax()]

print("\nBest Model:", best_model['Model'])
print("Best R2 Score:", best_model['R2 Score'])
print("MAE:", best_model['MAE'])
print("RMSE:", best_model['RMSE'])
