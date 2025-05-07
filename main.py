import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import logging  # For better output management

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
RANDOM_STATE = 42  # Define a constant for consistent random state

def preprocess_data(df, output_pdf_path="preprocessing_plots.pdf"):
    
    logging.info("Starting Data Preprocessing...")
    df_processed = df.copy()

    with PdfPages(output_pdf_path) as pdf:
        logging.info("\n1. Data Type Conversion:")
        if 'timestamp' in df_processed.columns:
            df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], errors='coerce')
            logging.info("  - 'timestamp' converted to datetime.")

        cols_to_convert = ['equipment_energy_consumption', 'lighting_energy',
                            'zone1_temperature', 'zone1_humidity', 'zone2_temperature']
        converted_cols_summary = []
        for col in cols_to_convert:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                converted_cols_summary.append(col)
        if converted_cols_summary:
            logging.info(f"  - Columns {converted_cols_summary} converted to numeric (errors coerced to NaN).")

        logging.info("\n2. Missing Value Handling:")
        filled_cols_summary = []
        for col in df_processed.columns:
            if df_processed[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    mean_val = df_processed[col].mean()
                    df_processed[col] = df_processed[col].fillna(mean_val)
                    if not pd.isna(mean_val):
                        filled_cols_summary.append(f"{col} (mean: {mean_val:.2f})")
                    logging.info(
                        f"  - Missing values in '{col}' imputed with the mean ({mean_val:.2f}). "
                        "This is a simple and effective approach when missingness is assumed to be "
                        "Missing Completely At Random (MCAR).  For Missing At Random (MAR) or "
                        "Missing Not At Random (MNAR) scenarios, more sophisticated imputation "
                        "techniques (e.g., multiple imputation, KNN imputation) might be considered."
                    )
                else:
                    # For non-numeric columns, fill with the most frequent value (mode)
                    mode_val = df_processed[col].mode()[0]
                    df_processed[col] = df_processed[col].fillna(mode_val)
                    logging.info(
                        f"  - Missing values in non-numeric column '{col}' imputed with the mode "
                        f"('{mode_val}')."
                    )
        if filled_cols_summary:
            logging.info(f"  - Missing numeric values filled with mean in: {', '.join(filled_cols_summary)}.")
        else:
            logging.info("  - No missing values found or imputed.")

        logging.info("\n3. Outlier Removal using IQR and Plotting:")
        cols_to_check_outliers = ['equipment_energy_consumption', 'lighting_energy',
                                    'zone1_temperature', 'zone1_humidity', 'zone2_temperature',
                                    'outdoor_temperature', 'atmospheric_pressure', 'outdoor_humidity',
                                    'wind_speed', 'visibility_index', 'dew_point',
                                    'random_variable1', 'random_variable2']
        outliers_removed_count = 0
        for col in cols_to_check_outliers:
            if col not in df_processed.columns or not pd.api.types.is_numeric_dtype(
                    df_processed[col]) or df_processed[col].isnull().all():
                continue

            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:
                logging.warning(f"  - IQR is 0 for column '{col}'. Skipping outlier removal for this column.")
                continue

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df_processed[
                (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)]
            num_outliers = outliers.shape[0]

            if num_outliers > 0:
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                sns.boxplot(y=df_processed[col], color='blue')
                plt.title(f'{col} Before Outlier Removal', fontsize=10)
                if not outliers.empty:
                    plt.scatter(x=np.zeros(len(outliers)), y=outliers[col], color='red', s=30,
                                zorder=3)

                df_processed = df_processed[
                    (df_processed[col] >= lower_bound) & (df_processed[col] <= upper_bound)]

                plt.subplot(1, 2, 2)
                sns.boxplot(y=df_processed[col], color='green')
                plt.title(f'{col} After Outlier Removal', fontsize=10)
                plt.tight_layout()
                pdf.savefig()
                plt.show()
                plt.close()
                logging.info(
                    f"  - Outliers removed from '{col}' ({num_outliers} data points) using IQR "
                    f"(Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}). "
                    "The IQR method is robust to some degree, but it's important to visualize "
                    "the data and consider the domain context.  Removing a large percentage of data "
                    "may indicate that the data generation process has changed."
                )
                outliers_removed_count += 1
            else:
                logging.info(f"  - No outliers found in '{col}' based on IQR criteria.")
        if outliers_removed_count == 0:
            logging.info("  - No outliers found or removed based on IQR criteria in checked columns.")

        logging.info("\n4. Data Binning for Temperature:")
        temp_cols_to_bin = [f'zone{i}_temperature' for i in range(1, 10)] + [
            'outdoor_temperature']
        available_temp_cols = [col for col in temp_cols_to_bin if
                               col in df_processed.columns and pd.api.types.is_numeric_dtype(
                                   df_processed[col]) and not df_processed[col].isnull().all()]

        if available_temp_cols:
            all_temps = pd.concat([df_processed[col] for col in available_temp_cols]).dropna()
            if not all_temps.empty:
                min_temp, max_temp = all_temps.min(), all_temps.max()
                bins = sorted(list(set([min_temp, 18, 25, max_temp])))
                if len(bins) < 2 or bins[0] == bins[-1]:
                    logging.warning(
                        f"  - Temperature range [{min_temp:.1f}-{max_temp:.1f}] not suitable for distinct "
                        "binning with 18, 25. Skipping."
                    )
                else:
                    labels = ['low', 'mid', 'high'][:len(bins) - 1] if len(bins) > 2 else [
                        'range1'] if len(bins) == 2 else []
                    if not labels:
                        labels = ['default_cat']

                    binned_temp_cols_count = 0
                    for col in available_temp_cols:
                        df_processed[col + '_bin'] = pd.cut(df_processed[col], bins=bins,
                                                            labels=labels, include_lowest=True,
                                                            duplicates='drop')
                        binned_temp_cols_count += 1
                        # Consider plotting bin distribution here
                    logging.info(
                        f"  - {binned_temp_cols_count} temperature columns binned into categories: "
                        f"'{', '.join(labels)}' using overall range [{min_temp:.1f}-{max_temp:.1f}] and "
                        "thresholds 18, 25.  Binning can help capture non-linear relationships and simplify "
                        "the model, but choice of bins should be informed by domain knowledge."
                    )
            else:
                logging.warning("  - No valid temperature data for defining overall min/max for binning.")
        else:
            logging.info("  - No suitable temperature columns found for binning.")

    logging.info(f"Preprocessing plots saved to '{output_pdf_path}'")
    logging.info("Preprocessing Done!")
    return df_processed


def perform_eda(df, output_pdf_path="eda_plots.pdf"):
    """
    Performs EDA and saves plots, providing detailed insights and interpretations.
    """
    logging.info("\nStarting Exploratory Data Analysis (EDA)...")
    df_eda = df.copy()
    eda_plots_generated = 0

    with PdfPages(output_pdf_path) as pdf:
        # Time Series Plot
        if 'timestamp' in df_eda.columns and 'equipment_energy_consumption' in df_eda.columns and \
                pd.api.types.is_datetime64_any_dtype(df_eda['timestamp']):

            temp_df_plot = df_eda.dropna(subset=['timestamp', 'equipment_energy_consumption']).sort_values(
                by='timestamp')
            if not temp_df_plot.empty:
                plt.figure(figsize=(15, 6))
                plt.plot(temp_df_plot['timestamp'],
                         temp_df_plot['equipment_energy_consumption'],
                         label='Energy Consumption')
                plt.title('Energy Consumption Over Time')
                plt.xlabel('Timestamp')
                plt.ylabel('Energy Consumption')
                plt.legend()
                plt.grid(True)
                pdf.savefig()
                plt.show()
                plt.close()
                logging.info(
                    "  - Time series plot of energy consumption generated.  This plot helps visualize trends, "
                    "seasonality, and potential anomalies in energy consumption over time.  For example, "
                    "we can look for increasing or decreasing trends, daily or weekly patterns, and sudden spikes "
                    "or drops in consumption."
                )
                eda_plots_generated += 1

                # Seasonal Decomposition
                df_decomp_src = temp_df_plot.set_index('timestamp')['equipment_energy_consumption'].dropna()
                if len(df_decomp_src) >= 2 * 24:  # Assuming period=24
                    try:
                        if not df_decomp_src.index.is_monotonic_increasing:
                            df_decomp_src = df_decomp_src.sort_index()
                        decomposition = sm.tsa.seasonal_decompose(df_decomp_src, model='additive',
                                                                period=24,
                                                                extrapolate_trend='freq')
                        fig = decomposition.plot()
                        fig.set_size_inches(15, 10)
                        plt.suptitle('Seasonal Decomposition of Energy Consumption', y=1.02)
                        plt.tight_layout(rect=[0, 0, 1, 0.98])
                        pdf.savefig()
                        plt.show()
                        plt.close()
                        logging.info(
                            "  - Seasonal decomposition of energy consumption performed (assuming 24-hour "
                            "seasonality). This decomposes the time series into trend, seasonality, and "
                            "residual components. The trend shows the long-term direction of energy consumption, "
                            "seasonality reveals recurring patterns within a day, and the residual represents "
                            "unexplained variations.  For example, we might see a trend of increasing energy "
                            "consumption over time, a peak in consumption during daytime hours, and random "
                            "fluctuations due to weather or other factors."
                        )
                        eda_plots_generated += 1
                    except Exception as e:
                        logging.warning(
                            f"  - Note: Seasonal decomposition failed or was skipped ({e}).  Ensure sufficient "
                            "data points and uniform frequency in the time series.  Consider resampling the data."
                        )

        # Correlation Heatmap
        numeric_df = df_eda.select_dtypes(include=np.number)
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Heatmap')
            pdf.savefig()
            plt.show()
            plt.close()
            logging.info(
                "  - Correlation heatmap generated. This visualizes the linear relationships between "
                "numerical features.  Values range from -1 (perfect negative correlation) to +1 (perfect "
                "positive correlation), with 0 indicating no linear correlation.  High correlations "
                "between features and the target variable (equipment_energy_consumption) suggest "
                "potential predictors, while high correlations between predictor features themselves "
                "might indicate multicollinearity, which can affect model stability."
            )
            eda_plots_generated += 1

            # Add insights from the correlation matrix.
            if 'equipment_energy_consumption' in corr_matrix:
                target_correlations = corr_matrix['equipment_energy_consumption'].sort_values(
                    ascending=False)
                top_positive_corr = target_correlations[1:6]  # Exclude itself
                top_negative_corr = target_correlations.sort_values().head(5)

                logging.info(
                    f"  - Top 5 positive correlations with energy consumption: "
                    f"{', '.join([f'{col} ({val:.2f})' for col, val in top_positive_corr.items()])}"
                )
                logging.info(
                    f"  - Top 5 negative correlations with energy consumption: "
                    f"{', '.join([f'{col} ({val:.2f})' for col, val in top_negative_corr.items()])}"
                )
                # Check for potential multicollinearity
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        if abs(corr_matrix.iloc[i, j]) > 0.8:  # Threshold for high correlation
                            high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
                if high_corr_pairs:
                    logging.warning(
                        f"  - Potential multicollinearity detected. The following pairs of features have a "
                        f"correlation greater than 0.8: {high_corr_pairs}.  Consider addressing this with "
                        "feature selection, PCA, or regularization."
                    )
                else:
                    logging.info("   - No significant multicollinearity detected (correlation < 0.8).")

        # Key Scatter Plots
        if 'equipment_energy_consumption' in df_eda:
            for col_scatter in ['lighting_energy', 'zone1_temperature',
                                'outdoor_temperature']:
                if col_scatter in df_eda and pd.api.types.is_numeric_dtype(
                        df_eda[col_scatter]):
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(x=col_scatter, y='equipment_energy_consumption',
                                    data=df_eda)
                    plt.title(f'Energy Consumption vs {col_scatter}')
                    pdf.savefig()
                    plt.show()
                    plt.close()
                    logging.info(
                        f"  - Scatter plot of Energy Consumption vs {col_scatter} generated.  These plots "
                        "help visualize the relationship between the predictor variables and the target "
                        "variable.  We can look for linear or non-linear patterns, and whether the variability "
                        "of energy consumption changes with the predictor variable (heteroscedasticity)."
                    )
                    eda_plots_generated += 1

    logging.info(f"  - {eda_plots_generated} EDA plots generated and saved to '{output_pdf_path}'.")
    logging.info("EDA Done!")
    return df


def perform_feature_engineering(df):
    """
    Performs feature engineering with detailed justifications for created features and handles edge cases.
    """
    logging.info("\nStarting Feature Engineering...")
    df_eng = df.copy()
    features_created_summary = []

    if 'timestamp' in df_eng.columns and pd.api.types.is_datetime64_any_dtype(
            df_eng['timestamp']):
        df_eng['hour_of_day'] = df_eng['timestamp'].dt.hour
        df_eng['day_of_week'] = df_eng['timestamp'].dt.dayofweek
        df_eng['month'] = df_eng['timestamp'].dt.month
        features_created_summary.append("time-based (hour, day, month)")
        logging.info(
            "  - Created time-based features (hour_of_day, day_of_week, month) to capture potential "
            "temporal patterns in energy consumption.  For example, energy consumption might be higher "
            "during certain hours of the day, on weekdays compared to weekends, or in specific months."
        )

        temp_cols = [f'zone{i}_temperature' for i in range(1, 10)] + ['outdoor_temperature']
        interaction_created_flag = False
        for temp_col in temp_cols:
            if temp_col in df_eng.columns and pd.api.types.is_numeric_dtype(
                    df_eng[temp_col]):
                df_eng[f'{temp_col}_x_hour'] = df_eng[temp_col] * df_eng['hour_of_day']
                interaction_created_flag = True
        if interaction_created_flag:
            features_created_summary.append("temperature-hour interactions")
            logging.info(
                "  - Created interaction features between temperature and hour_of_day to model how the "
                "effect of temperature on energy consumption might vary throughout the day.  For instance, "
                "the impact of outdoor temperature on energy consumption might be different during the day "
                "when cooling systems are used more heavily compared to at night."
            )

    if 'equipment_energy_consumption' in df_eng.columns and pd.api.types.is_numeric_dtype(
            df_eng['equipment_energy_consumption']):
        df_eng['energy_lag_24'] = df_eng['equipment_energy_consumption'].shift(24)
        df_eng['energy_rolling_24'] = df_eng['equipment_energy_consumption'].rolling(
            window=24, min_periods=1).mean()
        # fill the NaNs
        for col_fill in ['energy_lag_24', 'energy_rolling_24']:
            if col_fill in df_eng and df_eng[col_fill].isnull().any():
                df_eng[col_fill] = df_eng[col_fill].fillna(df_eng[col_fill].mean())
        features_created_summary.append("lagged/rolling energy features")
        logging.info(
            "  - Created lagged (24-hour) and rolling (24-hour mean) energy consumption features.  "
            "The lagged feature captures the energy consumption from the same time yesterday, which can "
            "be useful for modeling daily patterns.  The rolling mean smooths out short-term fluctuations "
            "and provides a measure of recent energy consumption trends."
        )

    if 'timestamp' in df_eng.columns:
        df_eng = df_eng.drop(columns=['timestamp'])  # Drop original timestamp

    binned_cols_converted = 0
    for col in df_eng.columns:  # Convert binned categories to codes
        if df_eng[col].dtype.name == 'category' and '_bin' in col:
            df_eng[col] = df_eng[col].cat.codes
            binned_cols_converted += 1
    if binned_cols_converted > 0:
        features_created_summary.append(f"{binned_cols_converted} binned cols to codes")
        logging.info(
            f"  - Converted {binned_cols_converted} binned categorical columns to numerical codes using "
            "pd.Categorical.codes.  This is necessary for using these features in most machine learning models."
        )
    # fill the remaining NaNs
    df_eng = df_eng.fillna(df_eng.mean(numeric_only=True))
    logging.info("  - Remaining NaN values in numeric columns (if any) imputed with the column mean.")

    if features_created_summary:
        logging.info(f"  - Created features: {'; '.join(features_created_summary)}.")
    else:
        logging.info("  - No new features were explicitly created in this step.")
    logging.info("Feature Engineering Done!")
    return df_eng



def perform_feature_selection(df, target_column='equipment_energy_consumption',
                             output_pdf_path="feature_selection_plots.pdf"):
    """
    Performs feature selection using Permutation Importance with detailed explanation and handling of edge cases.
    """
    logging.info("\nStarting Feature Selection...")
    if target_column not in df.columns:
        logging.error(
            f"  - Target column '{target_column}' not found in DataFrame.  Feature selection cannot be performed."
            "  Returning the original dataframe."
        )
        return df  # Return original DataFrame to avoid errors later

    X = df.drop(columns=[target_column], errors='ignore')
    y = df[target_column]

    # Ensure X is numeric and handle NaNs
    X = X.select_dtypes(include=np.number).dropna(axis=1,
                                                    how='all')  # Keep only numeric, non-all-NaN columns
    X = X.fillna(X.mean())
    if X.empty:
        logging.error(
            "  - No numeric features available after selection and NaN handling.  Feature selection cannot be performed."
            "  Returning the original dataframe."
        )
        return df

    y = y.reindex(X.index)  # Align y with X's index
    if y.isnull().any():
        y = y.fillna(y.mean())  # Fill NaNs in y

    X_train, X_test_perm, y_train, y_test_perm = train_test_split(X, y, test_size=0.2,
                                                                random_state=RANDOM_STATE)
    if X_train.empty or X_test_perm.empty:
        logging.error(
            "  - Not enough data for feature selection split (train or test set is empty).  Feature selection"
            " cannot be performed.  Returning the original dataframe."
        )
        return df

    with PdfPages(output_pdf_path) as pdf:
        model_rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1,
                                            n_estimators=50)  # Reduced estimators for speed
        model_rf.fit(X_train, y_train)
        result = permutation_importance(model_rf, X_test_perm, y_test_perm,
                                            n_repeats=5, random_state=RANDOM_STATE,
                                            n_jobs=-1, scoring='r2')

        feature_importance_df = pd.DataFrame(
            {'Feature': X.columns, 'Importance': result.importances_mean}).sort_values(
            by='Importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
        plt.title('Top 15 Feature Importances (PSA-RF)')
        plt.tight_layout()
        pdf.savefig()
        plt.show()
        plt.close()
        logging.info(
            "  - Permutation Importance using Random Forest (PSA-RF) was used for feature selection.  "
            "This method measures the decrease in model performance (R^2) when a feature's values are "
            "randomly shuffled.  Features with higher importance scores are more influential in the model's "
            "predictions.  The top 15 most important features are visualized in the plot."
        )

        selected_features = feature_importance_df[feature_importance_df['Importance'] > 0.001]['Feature'].tolist()
        if not selected_features:
            top_n = min(10, len(X.columns))
            selected_features = feature_importance_df['Feature'].head(
                top_n).tolist()  # Select top N features
            logging.warning(
                f"  - No features met importance threshold > 0.001.  Selected top {top_n} features based on "
                "permutation importance.  This can happen if all features have weak predictive power."
            )
        else:
            logging.info(
                f"  - Selected {len(selected_features)} features via PSA-RF: {selected_features}"
            )

    logging.info(f"Feature selection plots saved to '{output_pdf_path}'")
    logging.info("Feature Selection Done!")
    return X[selected_features]



def train_and_evaluate_model(X, y, model_type='random_forest'):
    """
    Trains and evaluates a regression model (Random Forest or Gradient Boosting) using cross-validation.
    Prints detailed results, including cross-validation performance and test set performance.
    """
    logging.info(f"\n--- Training and Evaluating: {model_type} ---")

    # Use the existing train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    if X_train.empty or X_test.empty:
        logging.error(f"  - Not enough data for training/testing {model_type}. Skipping model.")
        return None, None  # Return None, None to indicate failure

    if model_type == 'random_forest':
        base_model = RandomForestRegressor(random_state=RANDOM_STATE)
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None],
                      'min_samples_split': [2, 4, 8]}  # Expanded grid
    elif model_type == 'gradient_boosting':
        base_model = GradientBoostingRegressor(random_state=RANDOM_STATE)
        param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.05, 0.1],
                      'max_depth': [3, 5, 7]}  # Expanded grid
    else:
        raise ValueError(
            f"Invalid model_type: {model_type}. Must be 'random_forest' or 'gradient_boosting'.")

    # Perform cross-validation *before* grid search to get a baseline
    logging.info("  - Performing cross-validation on the base model (before tuning):")
    cv_results_before = cross_validate_model(base_model, X_train, y_train)

    # Grid Search for hyperparameter tuning
    logging.info("  - Performing Grid Search for hyperparameter tuning...")
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='r2',
                                verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logging.info(f"  - Best Hyperparameters for {model_type}: {grid_search.best_params_}")

    # Perform cross-validation on the best model
    logging.info("  - Performing cross-validation on the best model (after tuning):")
    cv_results_after = cross_validate_model(best_model, X_train, y_train)

    # Evaluate on the test set
    y_pred_test = best_model.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)

    logging.info(f"\n  - {model_type.capitalize()} Model Evaluation Results:")
    logging.info(
        f"    - Cross-Validation Results (Before Tuning): R2 (mean): {cv_results_before['r2_mean']:.4f}, MSE (mean): {cv_results_before['mse_mean']:.2f}")
    logging.info(
        f"    - Cross-Validation Results (After Tuning):  R2 (mean): {cv_results_after['r2_mean']:.4f}, MSE (mean): {cv_results_after['mse_mean']:.2f}")
    logging.info(f"    - Test Set Performance:")
    logging.info(f"      - RMSE: {rmse_test:.2f}")
    logging.info(f"      - R2 Score: {r2_test:.4f}")
    logging.info(f"      - MSE: {mse_test:.2f}")
    logging.info(f"      - MAE: {mae_test:.2f}")

    return best_model, {'r2': r2_test, 'mse': mse_test, 'mae': mae_test, 'rmse': rmse_test} 



def cross_validate_model(model, X, y, cv=5):
    """
    Performs cross-validation on a given model and returns the mean R2, MSE, and RMSE.

    Args:
        model: The sklearn model to cross-validate.
        X: The input features.
        y: The target variable.
        cv: The number of cross-validation folds (default: 5).

    Returns:
        A dictionary containing the mean R2, MSE, and RMSE across the folds.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    r2_scores = []
    mse_scores = []
    rmse_scores = []  # Added RMSE scores

    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_val_fold)

        r2_scores.append(r2_score(y_val_fold, y_pred_fold))
        mse_scores.append(mean_squared_error(y_val_fold, y_pred_fold))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_pred_fold)))  # Calculate RMSE

    return {'r2_mean': np.mean(r2_scores), 'mse_mean': np.mean(mse_scores),
            'rmse_mean': np.mean(rmse_scores)}  # Return mean RMSE

def main():
    """
    Main pipeline function with detailed logging and error handling.
    """
    try:
        # Ensure 'data/data.csv' is in a 'data' subdirectory relative to the script, or provide an absolute path.
        df_raw = pd.read_csv('data/data.csv')
        logging.info("Data loaded successfully!")
    except FileNotFoundError:
        logging.error(
            "Error: 'data/data.csv' not found. Please check the path or ensure the file is in a 'data' "
            "subdirectory."
        )
        return
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return
    logging.info(f"Dataset info:\n{df_raw.info()}")
    logging.info(f"Dataset shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    logging.info(f"Dataset summary statistics:\n{df_raw.describe()}")

    # For a very concise run, PDF generation can be made optional or suppressed.
    output_plots_to_pdf = True
    pdf_preprocessing = "preprocessing_plots.pdf" if output_plots_to_pdf else None
    pdf_eda = "eda_plots.pdf" if output_plots_to_pdf else None
    pdf_fs = "feature_selection_plots.pdf" if output_plots_to_pdf else None

    df_preprocessed = preprocess_data(df_raw, pdf_preprocessing)

    # The EDA function in the original script could modify df by setting index.
    # For a cleaner pipeline, it's better if EDA doesn't modify the df passed to subsequent stages,
    # or if it does, it returns the modified df explicitly.
    # Current EDA function returns original df reference passed.
    perform_eda(df_preprocessed, pdf_eda)

    df_engineered = perform_feature_engineering(df_preprocessed)

    target_col = 'equipment_energy_consumption'
    if target_col not in df_engineered.columns:
        logging.critical(
            f"Critical Error: Target column '{target_col}' missing after feature engineering. Halting."
        )
        return

    df_selected_features_X = perform_feature_selection(df_engineered,
                                                        target_column=target_col,
                                                        output_pdf_path=pdf_fs)
    if df_selected_features_X.empty:
        logging.critical("Error: Feature selection resulted in no features. Halting.")
        return

    X = df_selected_features_X
    # Align y with X's index (important if feature selection or prior steps altered X's index)
    y = df_engineered.loc[X.index, target_col].dropna()  # Ensure y is clean and aligned
    X = X.reindex(y.index)  # Final alignment

    if X.empty or y.empty or len(X) != len(y):
        logging.critical(
            f"Error: X ({X.shape}) or y ({y.shape}) is empty or misaligned after selection/alignment. Halting."
        )
        return

    logging.info(f"Final dataset for modeling: X shape {X.shape}, y shape {y.shape}")

    # Train and evaluate models.  Capture the return values.
    rf_model, rf_results = train_and_evaluate_model(X.copy(), y.copy(),
                                                    model_type='random_forest')
    gb_model, gb_results = train_and_evaluate_model(X.copy(), y.copy(),
                                                    model_type='gradient_boosting')

    logging.info("\n--- Final Model Performance Comparison ---")
    best_overall_model_name = "N/A"
    best_r2_overall = -float('inf')
    best_rmse_overall = float('inf')  # Initialize with positive infinity

    if rf_model and rf_results:  # Check if rf_model was successfully trained
        logging.info(f"Random Forest (Tuned) R2 Score: {rf_results['r2']:.4f}, RMSE: {rf_results['rmse']:.2f}, MSE: {rf_results['mse']:.2f}")
        if (rf_results['r2'] > best_r2_overall) or \
           (rf_results['r2'] == best_r2_overall and rf_results['rmse'] < best_rmse_overall):
            best_r2_overall = rf_results['r2']
            best_rmse_overall = rf_results['rmse']
            best_overall_model_name = "Random Forest"
    else:
        logging.warning("Random Forest model training failed or was skipped.")

    if gb_model and gb_results:  # Check if gb_model was successfully trained
        logging.info(f"Gradient Boosting (Tuned) R2 Score: {gb_results['r2']:.4f}, RMSE: {gb_results['rmse']:.2f}, MSE: {gb_results['mse']:.2f}")
        if (gb_results['r2'] > best_r2_overall) or \
           (gb_results['r2'] == best_r2_overall and gb_results['rmse'] < best_rmse_overall):
            best_r2_overall = gb_results['r2']
            best_rmse_overall = gb_results['rmse']
            best_overall_model_name = "Gradient Boosting"
        elif rf_model and rf_results and gb_results['r2'] == best_r2_overall and gb_results['rmse'] == best_rmse_overall:
            best_overall_model_name = "Random Forest & Gradient Boosting (similar performance)"

    if best_overall_model_name != "N/A":
        logging.info(
            f"\nBest Performing Model: {best_overall_model_name} with R2 Score: {best_r2_overall:.4f} and RMSE: {best_rmse_overall:.2f}")
    else:
        logging.warning("\nNo models were successfully trained and evaluated.")

    logging.info("\nScript execution finished.")

if __name__ == "__main__":
    main()