import pandas as pd


class GroupEstimate:
    def __init__(self, estimate):
        """
        Initialize GroupEstimate with the specified estimation method.
        
        Parameters:
        estimate : str
        
        Must be either 'mean' or 'median' for method to work
        """
        if estimate not in ['mean', 'median']:
            raise ValueError("estimate must be either 'mean' or 'median'")
        self.estimate = estimate
        self.group_estimates = None
    
    def fit(self, X, y):
        """
        Fit the model by calculating group estimates from the training data.
        
        Parameters:
        X: pandas.DataFrame: DataFrame of categorical features
        y : array-like: Target values
            
        Returns self for method chaining
        """
        # Convert input to df
        X = pd.DataFrame(X).copy()
        
        # Combine X and y into a single df
        data = X.copy()
        data['y'] = y
        
        # Group and calculate means and medians
        if self.estimate == 'mean':
            self.group_estimates = data.groupby(list(X.columns))['y'].mean()
        else:
            self.group_estimates = data.groupby(list(X.columns))['y'].median()
            
        return self

    def predict(self, X):
        """
        Predict using the fitted model.
        
        Parameters:
        X: pandas.DataFrame: Input data to predict on. Must have the same columns as the data used for fitting.
            
        Returns
        numpy.ndarray: Predicted values based on group estimates
            
        Raises: ValueError if the model has not been fitted yet
        """
        if self.group_estimates is None:
            raise ValueError("This GroupEstimate instance is not fitted yet. Call fit first")

        X = pd.DataFrame(X).copy()
        
        # Create a MultiIndex from the input data
        X_index = pd.MultiIndex.from_frame(X)
        
        # Map each group to its estimate
        predictions = X_index.map(self.group_estimates.get)
        
        # Check for any groups that weren't seen during training
        n_missing = predictions.isna().sum()
        if n_missing > 0:
            print(f"Warning: {n_missing} had N/A values. Returning N/A for them")
            
        return predictions.values