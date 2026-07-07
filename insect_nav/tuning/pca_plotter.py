import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class PCAPlotter:
    """Class for generating PCA-related plots."""

    def __init__(self, output_path, df_populations, imposed_params):
        """
        Initialize the PCAPlotter.

        Args:
            output_path (str or Path): Base path where plots will be saved.
                                      A "Pca" subfolder will be created inside this path.
            df_populations (pd.DataFrame): DataFrame containing all population data.
            imposed_params (list): List of column names for imposed parameters.
        """
        self.output_path = Path(output_path)
        self.pca_output_path = self.output_path / "Pca"
        self.pca_output_path.mkdir(parents=True, exist_ok=True)

        self.df_populations = df_populations
        self.imposed_params = imposed_params

    def _get_available_params(self):
        """
        Helper method to get available parameters that exist in the dataframe.

        Returns:
            list: Available imposed parameters.
        """
        return [p for p in self.imposed_params if p in self.df_populations.columns]

    def _prepare_pca_data(self, n_components):
        """
        Helper method to extract, normalize, and compute PCA.

        Args:
            n_components (int): Number of components to compute.

        Returns:
            tuple: (X_scaled, X_pca, pca, available_params) or (None, None, None, None) if insufficient data.
        """
        available_params = self._get_available_params()

        if not available_params or len(available_params) < n_components:
            print(f"Warning: Need at least {n_components} imposed parameters for PCA. Found: {len(available_params)}")
            return None, None, None, None

        # Extract and normalize imposed parameters
        X = self.df_populations[available_params].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        return X_scaled, X_pca, pca, available_params

    def plot_all_individuals_2d(self):
        """
        Plot all individuals from all generations in 2D PCA space.

        Points are colored by error (MAE). Saves the plot as 'all_individuals_2d.png'.
        """
        X_scaled, X_pca, pca, available_params = self._prepare_pca_data(n_components=2)

        if X_pca is None:
            print("Warning: No data available for PCA 2D all individuals")
            return

        # Get error and normalize for coloring
        errors = self.df_populations["Error"].values
        error_normalized = (errors - errors.min()) / (errors.max() - errors.min() + 1e-6)

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot all points colored by error
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=error_normalized,
                           cmap='viridis', s=50, alpha=0.6, edgecolors='black',
                           linewidth=0.5)

        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Error (MAE)", fontsize=11)

        # Calculate explained variance
        var_pc1 = pca.explained_variance_ratio_[0] * 100
        var_pc2 = pca.explained_variance_ratio_[1] * 100
        var_total = var_pc1 + var_pc2

        ax.set_xlabel(f"PC1 ({var_pc1:.1f}%)", fontsize=12)
        ax.set_ylabel(f"PC2 ({var_pc2:.1f}%)", fontsize=12)
        ax.set_title(f"All Individuals PCA 2D (Total Variance: {var_total:.1f}%)",
                    fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # Save the plot
        plot_path = self.pca_output_path / "all_individuals_2d.png"
        fig.savefig(plot_path, dpi=300)
        plt.close()

        print(f"Plot saved to {plot_path}")

    def plot_all_individuals_3d(self):
        """
        Plot all individuals from all generations in 3D PCA space.

        Points are colored by error (MAE). Saves the plot as 'all_individuals_3d.png'.
        """
        X_scaled, X_pca, pca, available_params = self._prepare_pca_data(n_components=3)

        if X_pca is None:
            print("Warning: No data available for PCA 3D all individuals")
            return

        try:
            # Import Axes3D here to avoid version conflicts
            from mpl_toolkits.mplot3d import Axes3D

            # Get error and normalize for coloring
            errors = self.df_populations["Error"].values
            error_normalized = (errors - errors.min()) / (errors.max() - errors.min() + 1e-6)

            # Create 3D plot
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')

            # Plot all points colored by error
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                               c=error_normalized, cmap='viridis', s=50, alpha=0.6,
                               edgecolors='black', linewidth=0.5)

            # Add colorbar
            cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
            cbar.set_label("Error (MAE)", fontsize=10)

            # Calculate explained variance
            var_pc1 = pca.explained_variance_ratio_[0] * 100
            var_pc2 = pca.explained_variance_ratio_[1] * 100
            var_pc3 = pca.explained_variance_ratio_[2] * 100
            var_total = var_pc1 + var_pc2 + var_pc3

            ax.set_xlabel(f"PC1 ({var_pc1:.1f}%)", fontsize=11)
            ax.set_ylabel(f"PC2 ({var_pc2:.1f}%)", fontsize=11)
            ax.set_zlabel(f"PC3 ({var_pc3:.1f}%)", fontsize=11)
            ax.set_title(f"All Individuals PCA 3D (Total Variance: {var_total:.1f}%)",
                        fontsize=14, fontweight="bold", pad=20)

            # Set viewing angle
            ax.view_init(elev=20, azim=45)

            fig.tight_layout()

            # Save the plot
            plot_path = self.pca_output_path / "all_individuals_3d.png"
            fig.savefig(plot_path, dpi=300)
            plt.close()

            print(f"Plot saved to {plot_path}")

        except (ImportError, ModuleNotFoundError) as e:
            print(f"Warning: Could not generate 3D PCA plot due to matplotlib version conflict: {e}")
            print("Skipping 3D PCA all individuals plot. The 2D version is available.")

    def plot_loadings_2d(self):
        """
        Plot the loadings (feature importance) for 2D PCA.

        Shows how each imposed parameter contributes to PC1 and PC2 using bar plots.
        Saves the plot as 'loadings_2d.png'.
        """
        X_scaled, X_pca, pca, available_params = self._prepare_pca_data(n_components=2)

        if pca is None:
            print("Warning: No data available for PCA 2D loadings")
            return

        # Get loadings (components)
        loadings = pca.components_.T

        # Create figure with 2 subplots (one for PC1, one for PC2)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Colors for bars
        colors = plt.cm.viridis([0.2, 0.8])

        # Plot PC1 loadings
        ax = axes[0]
        ax.bar(range(len(available_params)), loadings[:, 0], color=colors[0], alpha=0.7,
               edgecolor='black', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xticks(range(len(available_params)))
        ax.set_xticklabels(available_params, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel("Loading Value", fontsize=11)
        ax.set_title(f"PC1 Loadings ({pca.explained_variance_ratio_[0]*100:.1f}% variance)",
                    fontsize=12, fontweight="bold")
        ax.grid(True, axis='y', alpha=0.3)

        # Plot PC2 loadings
        ax = axes[1]
        ax.bar(range(len(available_params)), loadings[:, 1], color=colors[1], alpha=0.7,
               edgecolor='black', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xticks(range(len(available_params)))
        ax.set_xticklabels(available_params, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel("Loading Value", fontsize=11)
        ax.set_title(f"PC2 Loadings ({pca.explained_variance_ratio_[1]*100:.1f}% variance)",
                    fontsize=12, fontweight="bold")
        ax.grid(True, axis='y', alpha=0.3)

        fig.suptitle("PCA Feature Importance - 2D", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()

        # Save the plot
        plot_path = self.pca_output_path / "loadings_2d.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved to {plot_path}")

    def plot_loadings_3d(self):
        """
        Plot the loadings (feature importance) for 3D PCA.

        Shows how each imposed parameter contributes to PC1, PC2, and PC3 using bar plots.
        Saves the plot as 'loadings_3d.png'.
        """
        X_scaled, X_pca, pca, available_params = self._prepare_pca_data(n_components=3)

        if pca is None:
            print("Warning: No data available for PCA 3D loadings")
            return

        # Get loadings (components)
        loadings = pca.components_.T

        # Create figure with 3 subplots (one for PC1, PC2, PC3)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Colors for bars
        colors = plt.cm.viridis([0.2, 0.5, 0.8])

        # Plot PC1 loadings
        ax = axes[0]
        ax.bar(range(len(available_params)), loadings[:, 0], color=colors[0], alpha=0.7,
               edgecolor='black', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xticks(range(len(available_params)))
        ax.set_xticklabels(available_params, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel("Loading Value", fontsize=11)
        ax.set_title(f"PC1 Loadings ({pca.explained_variance_ratio_[0]*100:.1f}% variance)",
                    fontsize=12, fontweight="bold")
        ax.grid(True, axis='y', alpha=0.3)

        # Plot PC2 loadings
        ax = axes[1]
        ax.bar(range(len(available_params)), loadings[:, 1], color=colors[1], alpha=0.7,
               edgecolor='black', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xticks(range(len(available_params)))
        ax.set_xticklabels(available_params, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel("Loading Value", fontsize=11)
        ax.set_title(f"PC2 Loadings ({pca.explained_variance_ratio_[1]*100:.1f}% variance)",
                    fontsize=12, fontweight="bold")
        ax.grid(True, axis='y', alpha=0.3)

        # Plot PC3 loadings
        ax = axes[2]
        ax.bar(range(len(available_params)), loadings[:, 2], color=colors[2], alpha=0.7,
               edgecolor='black', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xticks(range(len(available_params)))
        ax.set_xticklabels(available_params, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel("Loading Value", fontsize=11)
        ax.set_title(f"PC3 Loadings ({pca.explained_variance_ratio_[2]*100:.1f}% variance)",
                    fontsize=12, fontweight="bold")
        ax.grid(True, axis='y', alpha=0.3)

        fig.suptitle("PCA Feature Importance - 3D", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()

        # Save the plot
        plot_path = self.pca_output_path / "loadings_3d.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved to {plot_path}")

    def plot_scree(self):
        """
        Plot the scree plot showing variance explained by each PC.

        Shows the first 10 principal components with their individual and
        cumulative explained variance ratios. Saves the plot as 'scree.png'.
        """
        available_params = self._get_available_params()

        if not available_params or len(available_params) < 2:
            print("Warning: Need at least 2 imposed parameters for PCA")
            return

        # Determine how many components to compute (min of 10 or number of available params)
        n_components = min(10, len(available_params))

        X_scaled, X_pca, pca, _ = self._prepare_pca_data(n_components=n_components)

        if pca is None:
            print("Warning: No data available for PCA scree plot")
            return

        # Get variance explained by each component
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot bar chart for individual variance
        pc_labels = [f"PC{i+1}" for i in range(n_components)]
        bars = ax.bar(pc_labels, explained_var, alpha=0.7, color='steelblue',
                      edgecolor='black', linewidth=1.5, label='Individual Variance')

        # Add percentage labels on bars
        for i, (bar, var) in enumerate(zip(bars, explained_var)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{var*100:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Plot cumulative variance on secondary axis
        ax2 = ax.twinx()
        line = ax2.plot(pc_labels, cumulative_var, color='red', marker='o',
                       linewidth=2.5, markersize=8, label='Cumulative Variance')
        ax2.set_ylabel("Cumulative Explained Variance", fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim([0, 1.05])

        # Add percentage labels on cumulative line
        for i, (label, cum_var) in enumerate(zip(pc_labels, cumulative_var)):
            ax2.text(i, cum_var + 0.02, f'{cum_var*100:.1f}%',
                    ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')

        # Labels and title
        ax.set_xlabel("Principal Components", fontsize=12)
        ax.set_ylabel("Explained Variance Ratio", fontsize=12)
        ax.set_title(f"Scree Plot - PCA Variance Explained (Total: {cumulative_var[-1]*100:.1f}%)",
                    fontsize=14, fontweight="bold")
        ax.grid(True, axis='y', alpha=0.3)

        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)

        fig.tight_layout()

        # Save the plot
        plot_path = self.pca_output_path / "scree.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved to {plot_path}")

    def plot_components_scatter_matrix(self):
        """
        Plot scatter plots for all pairs of principal components.

        Creates a scatter plot for each pair of components (PC1 vs PC2, PC1 vs PC3, etc.)
        with points colored by error. Uses the first 10 components (or fewer if not available).
        All plots are saved in a Scatter subfolder.

        Saves plots as 'component_pair_PCi_vs_PCj.png'.
        """
        available_params = self._get_available_params()

        if not available_params or len(available_params) < 2:
            print("Warning: Need at least 2 imposed parameters for PCA")
            return

        # Determine how many components to compute (min of 10 or number of available params)
        n_components = min(10, len(available_params))

        X_scaled, X_pca, pca, _ = self._prepare_pca_data(n_components=n_components)

        if X_pca is None:
            print("Warning: No data available for PCA component scatter plots")
            return

        # Create Scatter subfolder
        scatter_path = self.pca_output_path / "Scatter"
        scatter_path.mkdir(parents=True, exist_ok=True)

        # Get error and normalize for coloring
        errors = self.df_populations["Error"].values
        error_normalized = (errors - errors.min()) / (errors.max() - errors.min() + 1e-6)

        # Create scatter plots for all pairs of components
        pc_labels = [f"PC{i+1}" for i in range(n_components)]

        for i in range(n_components):
            for j in range(i + 1, n_components):
                fig, ax = plt.subplots(figsize=(10, 8))

                # Create scatter plot colored by error
                scatter = ax.scatter(X_pca[:, i], X_pca[:, j],
                                   c=error_normalized,
                                   cmap='viridis', s=50, alpha=0.6,
                                   edgecolors='black', linewidth=0.5)

                # Add colorbar
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.set_label("Error (MAE)", fontsize=11)

                # Calculate explained variance for these components
                var_pc_i = pca.explained_variance_ratio_[i] * 100
                var_pc_j = pca.explained_variance_ratio_[j] * 100

                ax.set_xlabel(f"{pc_labels[i]} ({var_pc_i:.1f}%)", fontsize=12)
                ax.set_ylabel(f"{pc_labels[j]} ({var_pc_j:.1f}%)", fontsize=12)
                ax.set_title(f"{pc_labels[i]} vs {pc_labels[j]} - All Individuals",
                            fontsize=14, fontweight="bold")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()

                # Save the plot
                plot_filename = f"component_pair_{pc_labels[i]}_vs_{pc_labels[j]}.png"
                plot_path = scatter_path / plot_filename
                fig.savefig(plot_path, dpi=300)
                plt.close()

                print(f"Plot saved to {plot_path}")

    def plot_components_error_correlation(self):
        """
        Plot the correlation between each principal component and the error.

        Creates a bar plot showing the Pearson correlation coefficient between
        each principal component and the network error. Uses the first 10 components
        (or fewer if not available).

        Saves the plot as 'components_error_correlation.png'.
        """
        available_params = self._get_available_params()

        if not available_params or len(available_params) < 2:
            print("Warning: Need at least 2 imposed parameters for PCA")
            return

        # Determine how many components to compute (min of 10 or number of available params)
        n_components = min(10, len(available_params))

        X_scaled, X_pca, pca, _ = self._prepare_pca_data(n_components=n_components)

        if X_pca is None:
            print("Warning: No data available for PCA error correlation plot")
            return

        # Get error values
        errors = self.df_populations["Error"].values

        # Calculate correlation between each component and error
        correlations = []
        for i in range(n_components):
            corr = np.corrcoef(X_pca[:, i], errors)[0, 1]
            correlations.append(corr)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create PC labels
        pc_labels = [f"PC{i+1}" for i in range(n_components)]

        # Create color map: blue for negative, red for positive
        colors = ['#1f77b4' if corr < 0 else '#d62728' for corr in correlations]

        # Plot bar chart
        bars = ax.bar(pc_labels, correlations, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=1.5)

        # Add correlation value labels on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{corr:.3f}',
                   ha='center', va='bottom' if corr > 0 else 'top',
                   fontsize=9, fontweight='bold')

        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        # Labels and title
        ax.set_xlabel("Principal Components", fontsize=12)
        ax.set_ylabel("Pearson Correlation Coefficient", fontsize=12)
        ax.set_title("Correlation between Principal Components and Error",
                    fontsize=14, fontweight="bold")
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim([min(correlations) - 0.15, max(correlations) + 0.15])

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', alpha=0.7, edgecolor='black', label='Negative Correlation'),
            Patch(facecolor='#d62728', alpha=0.7, edgecolor='black', label='Positive Correlation')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

        fig.tight_layout()

        # Save the plot
        plot_path = self.pca_output_path / "components_error_correlation.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved to {plot_path}")

    def plot_mae_prediction(self):
        """
        Plot the prediction error between predicted and actual MAE using linear regression.

        Uses all available principal components to predict the MAE.
        Creates a scatter plot of actual vs predicted MAE with R² score and regression line.

        Saves the plot as 'mae_prediction.png'.
        """
        available_params = self._get_available_params()

        if not available_params or len(available_params) < 2:
            print("Warning: Need at least 2 imposed parameters for PCA")
            return

        # Use all available components for regression
        n_components = len(available_params)

        X_scaled, X_pca, pca, _ = self._prepare_pca_data(n_components=n_components)

        if X_pca is None:
            print("Warning: No data available for MAE prediction plot")
            return

        # Get actual error values
        y_actual = self.df_populations["Error"].values

        # Fit linear regression
        model = LinearRegression()
        model.fit(X_pca, y_actual)
        y_predicted = model.predict(X_pca)

        # Calculate metrics
        r2 = r2_score(y_actual, y_predicted)
        mae = mean_absolute_error(y_actual, y_predicted)
        rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot of actual vs predicted
        scatter = ax.scatter(y_actual, y_predicted, alpha=0.6, s=50,
                           edgecolors='black', linewidth=0.5)

        # Regression line (perfect prediction)
        min_val = min(y_actual.min(), y_predicted.min())
        max_val = max(y_actual.max(), y_predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        # Labels and title
        ax.set_xlabel("Actual MAE", fontsize=12)
        ax.set_ylabel("Predicted MAE", fontsize=12)
        ax.set_title(f"MAE Prediction using Linear Regression on PCs\nR²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}",
                    fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

        fig.tight_layout()

        # Save the plot
        plot_path = self.pca_output_path / "mae_prediction.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved to {plot_path}")
        print(f"  - R² Score: {r2:.4f}")
        print(f"  - Mean Absolute Error: {mae:.4f}")
        print(f"  - Root Mean Squared Error: {rmse:.4f}")

    def plot_regression_coefficients(self):
        """
        Plot the coefficients of linear regression for MAE prediction.

        Shows the importance of each principal component in predicting MAE.
        Uses all available principal components.

        Saves the plot as 'regression_coefficients.png'.
        """
        available_params = self._get_available_params()

        if not available_params or len(available_params) < 2:
            print("Warning: Need at least 2 imposed parameters for PCA")
            return

        # Use all available components for regression
        n_components = len(available_params)

        X_scaled, X_pca, pca, _ = self._prepare_pca_data(n_components=n_components)

        if X_pca is None:
            print("Warning: No data available for regression coefficients plot")
            return

        # Get actual error values
        y_actual = self.df_populations["Error"].values

        # Fit linear regression
        model = LinearRegression()
        model.fit(X_pca, y_actual)
        coefficients = model.coef_
        intercept = model.intercept_

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create PC labels
        pc_labels = [f"PC{i+1}" for i in range(n_components)]

        # Create color map: blue for negative, red for positive
        colors = ['#1f77b4' if coef < 0 else '#d62728' for coef in coefficients]

        # Plot bar chart
        bars = ax.bar(pc_labels, coefficients, color=colors, alpha=0.7,
                      edgecolor='black', linewidth=1.5)

        # Add coefficient value labels on bars
        for bar, coef in zip(bars, coefficients):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{coef:.4f}',
                   ha='center', va='bottom' if coef > 0 else 'top',
                   fontsize=9, fontweight='bold')

        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        # Labels and title
        ax.set_xlabel("Principal Components", fontsize=12)
        ax.set_ylabel("Regression Coefficient", fontsize=12)
        ax.set_title(f"Linear Regression Coefficients for MAE Prediction (Intercept: {intercept:.4f})",
                    fontsize=14, fontweight="bold")
        ax.grid(True, axis='y', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', alpha=0.7, edgecolor='black', label='Negative Coefficient'),
            Patch(facecolor='#d62728', alpha=0.7, edgecolor='black', label='Positive Coefficient')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

        fig.tight_layout()

        # Save the plot
        plot_path = self.pca_output_path / "regression_coefficients.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved to {plot_path}")
        print(f"  - Intercept: {intercept:.4f}")
        print(f"  - Coefficients: {', '.join([f'a{i+1}={coef:.4f}' for i, coef in enumerate(coefficients)])}")

    def plot_all(self):
        """Generate all PCA plots."""
        self.plot_all_individuals_2d()
        self.plot_all_individuals_3d()
        self.plot_loadings_2d()
        self.plot_loadings_3d()
        self.plot_scree()
        self.plot_components_scatter_matrix()
        self.plot_components_error_correlation()
        self.plot_mae_prediction()
        self.plot_regression_coefficients()
