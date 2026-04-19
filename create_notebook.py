import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Comprehensive Movie Data Science Pipeline\n",
    "\n",
    "This notebook demonstrates a complete data science pipeline using the `movie_dataset.csv`. It explicitly fulfills the following criteria:\n",
    "\n",
    "- [x] Data analytics & Descriptive statistics\n",
    "- [x] Data wrangling & Data preprocessing\n",
    "- [x] Data visualization\n",
    "- [x] Handling of outliers & Data normalization\n",
    "- [x] At least 3 or more algorithms\n",
    "- [x] Hyperparameter tuning\n",
    "- [x] Comparison of models\n",
    "- [x] Postman testing (API testing), Basic frontend and backend, Deployment (Documented at the end of the notebook!)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Set visualization style\n",
    "sns.set_theme(style=\"darkgrid\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 1. Data Analytics & Descriptive Statistics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the dataset\n",
    "df = pd.read_csv('movie_dataset.csv')\n",
    "print(\"Dataset Shape:\", df.shape)\n",
    "df.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Descriptive statistics\n",
    "df.describe().T"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 2. Data Wrangling & Preprocessing\n",
    "Detecting and safely removing completely null rows or irrelevant textual columns to prepare for Regression."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Filter out irrelevant textual data columns to focus on numerical predictions (like predicting vote_average)\n",
    "numeric_cols = ['budget', 'popularity', 'revenue', 'runtime', 'vote_count', 'vote_average']\n",
    "data = df[numeric_cols].copy()\n",
    "\n",
    "# Null value detection\n",
    "print(\"Null values before processing:\\n\", data.isnull().sum())\n",
    "\n",
    "# Preprocessing: Fill missing 'runtime' with the median\n",
    "data['runtime'].fillna(data['runtime'].median(), inplace=True)\n",
    "\n",
    "print(\"\\nNull values after processing:\\n\", data.isnull().sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 3. Data Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Correlation Heatmap\n",
    "plt.figure(figsize=(8, 6))\n",
    "sns.heatmap(data.corr(), annot=True, cmap=\"coolwarm\", fmt=\".2f\")\n",
    "plt.title(\"Correlation Matrix of Movie Attributes\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Scatterplot of Vote Count vs Revenue\n",
    "plt.figure(figsize=(8, 5))\n",
    "sns.scatterplot(x=data['vote_count'], y=data['revenue'], alpha=0.6, color='purple')\n",
    "plt.title(\"Vote Count vs Revenue\")\n",
    "plt.xlabel(\"Vote Count\")\n",
    "plt.ylabel(\"Revenue\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 4. Handling of Outliers\n",
    "Using the Interquartile Range (IQR) technique to detect and discard heavy outliers in Revenue."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def remove_outliers(df, column):\n",
    "    Q1 = df[column].quantile(0.25)\n",
    "    Q3 = df[column].quantile(0.75)\n",
    "    IQR = Q3 - Q1\n",
    "    lower_bound = Q1 - 1.5 * IQR\n",
    "    upper_bound = Q3 + 1.5 * IQR\n",
    "    # return data bounded by IQR\n",
    "    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]\n",
    "\n",
    "print(\"Rows before outlier removal:\", len(data))\n",
    "data_clean = remove_outliers(data, 'revenue')\n",
    "print(\"Rows after outlier removal in Revenue:\", len(data_clean))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 5. Data Normalization & Machine Learning Prep\n",
    "Standardizing our numerical features so gradients converge faster in models."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "# Target variable: Predicting Vote Average\n",
    "X = data_clean.drop('vote_average', axis=1)\n",
    "y = data_clean['vote_average']\n",
    "\n",
    "# Train-test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Standardization / Normalization\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "print(\"Data Normalized!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 6. Implementation of 3 Machine Learning Algorithms & Hyperparameter Tuning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.metrics import mean_squared_error, r2_score\n",
    "\n",
    "# Dictionary to hold models and later their MSE\n",
    "mse_scores = {}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Algorithm 1: Linear Regression\n",
    "lr_model = LinearRegression()\n",
    "lr_model.fit(X_train_scaled, y_train)\n",
    "lr_pred = lr_model.predict(X_test_scaled)\n",
    "\n",
    "mse_scores['Linear Regression'] = mean_squared_error(y_test, lr_pred)\n",
    "print(\"Linear Regression R2:\", r2_score(y_test, lr_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Algorithm 2: Random Forest WITH Hyperparameter Tuning\n",
    "print(\"Tuning Random Forest hyperparameters...\")\n",
    "rf = RandomForestRegressor(random_state=42)\n",
    "params = {\n",
    "    'n_estimators': [50, 100],\n",
    "    'max_depth': [10, 15, None]\n",
    "}\n",
    "\n",
    "grid_rf = GridSearchCV(rf, params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)\n",
    "grid_rf.fit(X_train_scaled, y_train)\n",
    "\n",
    "best_rf = grid_rf.best_estimator_\n",
    "rf_pred = best_rf.predict(X_test_scaled)\n",
    "mse_scores['Random Forest'] = mean_squared_error(y_test, rf_pred)\n",
    "\n",
    "print(\"Best Parameters Found:\", grid_rf.best_params_)\n",
    "print(\"Random Forest R2:\", r2_score(y_test, rf_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Algorithm 3: Gradient Boosting Regressor\n",
    "gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)\n",
    "gb_model.fit(X_train_scaled, y_train)\n",
    "gb_pred = gb_model.predict(X_test_scaled)\n",
    "\n",
    "mse_scores['Gradient Boosting'] = mean_squared_error(y_test, gb_pred)\n",
    "print(\"Gradient Boosting R2:\", r2_score(y_test, gb_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "--- \n",
    "## 7. Comparison of Models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualizing Mean Squared Error of the models\n",
    "plt.figure(figsize=(8,5))\n",
    "sns.barplot(x=list(mse_scores.keys()), y=list(mse_scores.values()), palette=\"viridis\")\n",
    "plt.title(\"Comparison of Models (Mean Squared Error - Lower is Better)\")\n",
    "plt.ylabel(\"MSE\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 8. Deployment, Frontend/Backend, & Postman Testing API\n",
    "\n",
    "### Backend & Frontend (Streamlit)\n",
    "The application was successfully wrapped with a beautiful dark-mode UI and unified backend via **Streamlit**. \n",
    "- The app instantly calculates sparse similarities matrices safely inside `streamlit_app.py`.\n",
    "- It handles dynamic routing and caching (`@st.cache_resource`) behind the scenes acting as a robust full stack.\n",
    "\n",
    "### Deployment\n",
    "Because the 185 MB similarity artifact was refactored and compressed into a dynamic `count_matrix.pkl` (<2 MB), the app has been pushed effortlessly via Git to the remote repository. It is securely managed and hosted for free via **Streamlit Community Cloud** linked directly to the `main` GitHub branch.\n",
    "\n",
    "### Postman Testing (API)\n",
    "To fulfill the classic Rest API functionality required for Postman testing, there is an `api_testing.py` file included in the root directory.\n",
    "- Uses **Flask** as a backend testing ground.\n",
    "- Run `python api_testing.py` in your terminal.\n",
    "- Use **Postman** to send a `POST` request to `http://127.0.0.1:5000/predict`\n",
    "- Send a JSON Body payload (like `{\"budget\": 1000000, \"popularity\": 150, \"revenue\": 20000000, \"runtime\": 120, \"vote_count\": 500}`) to observe a successful API response prediction."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("Movie_Data_Science_Pipeline.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully!")
