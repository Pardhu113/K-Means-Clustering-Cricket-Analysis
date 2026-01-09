# K-Means Clustering Cricket Analysis

## Project Overview

This project performs K-Means clustering analysis on ODI (One Day International) cricket player performance statistics. The goal is to identify distinct groups or clusters of players based on their performance metrics using machine learning techniques.

## Dataset

- **Source**: ODI Cricket Player Statistics
- **Total Records**: 2,500+ cricket players
- **Features**: 13 performance metrics including:
  - Matches (mat)
  - Innings (inns)
  - Not Out (no)
  - Total Runs (runs)
  - Highest Score (hs)
  - Average (ave)
  - Balls Faced (bf)
  - Strike Rate (sr)
  - Centuries (100)
  - Half-centuries (50)
  - Ducks (0)
  - Experience (years in cricket)

## Key Features

✨ **Data Preprocessing**: Comprehensive data cleaning and handling missing values

🔄 **Standardization**: StandardScaler normalization for fair clustering

📊 **Elbow Method**: Optimal k-value determination

🎯 **K-Means Clustering**: Implementation with 3 optimal clusters

📈 **3D Visualization**: Interactive 3D scatter plots using Plotly

## Project Structure

```
K-Means-Clustering-Cricket-Analysis/
├── k_means_clustering.py       # Main clustering script
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/Pardhu113/K-Means-Clustering-Cricket-Analysis.git
cd K-Means-Clustering-Cricket-Analysis
```

2. **Install required packages**:
```bash
pip install -r requirements.txt
```

3. **Add your dataset**:
- Place your `ODI_data.csv` file in the project directory
- Ensure the CSV has proper column names for cricket statistics

## Usage

Run the clustering analysis:

```bash
python k_means_clustering.py
```

This will:
- Load and clean the cricket data
- Perform standardization
- Determine optimal number of clusters using Elbow Method
- Fit K-Means model with k=3
- Generate 3D visualization as HTML
- Save results and visualizations

## Results & Visualizations

### Output Files Generated:
- `elbow_curve.png` - Elbow curve showing inertia vs number of clusters
- `k_means_3d_visualization.html` - Interactive 3D scatter plot of clusters

### Cluster Analysis:
The model identifies 3 distinct player clusters:
- **Cluster 0**: High-performing, experienced players
- **Cluster 1**: Mid-tier performers with moderate experience
- **Cluster 2**: Emerging/developing players with limited statistics

## Technologies & Libraries

- **Python 3.8+**
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical data visualization
- **Plotly** - Interactive 3D visualizations

## Model Details

### K-Means Algorithm
- **Number of Clusters**: 3 (determined via Elbow Method)
- **Initialization**: k-means++ for better convergence
- **Random State**: 42 (for reproducibility)
- **Iterations**: Maximum 300

### Data Processing Pipeline
1. Data Loading (UTF-8 encoding with fallback to latin1)
2. Column Name Normalization
3. Missing Value Handling
4. Feature Selection (Numeric columns)
5. Standardization using StandardScaler
6. Clustering & Visualization

## Performance Metrics

- Inertia scores for different k values
- Cluster distribution and sizes
- Within-cluster sum of squares (WCSS)

## Future Enhancements

- [ ] Implement Silhouette Coefficient for cluster quality assessment
- [ ] Add Davies-Bouldin Index for cluster validation
- [ ] Develop interactive web dashboard for visualization
- [ ] Include PCA for dimensionality reduction
- [ ] Add clustering comparison (K-Means vs. Hierarchical vs. DBSCAN)
- [ ] Implement automated optimal k selection algorithms

## Learning Outcomes

This project demonstrates:
- Data preprocessing and cleaning techniques
- Feature scaling and normalization
- Unsupervised machine learning with K-Means
- Dimensionality reduction concepts
- Interactive data visualization
- Model evaluation and optimization

## References

- Scikit-learn K-Means Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- Plotly 3D Scatter: https://plotly.com/python/3d-scatter-plots/
- Cricket Statistics Data: Open cricket statistics databases

## Author

**Pardhasaradhi** - Data Science Enthusiast
- GitHub: [@Pardhu113](https://github.com/Pardhu113)
- Portfolio: [Project Repository](https://github.com/Pardhu113/K-Means-Clustering-Cricket-Analysis)

## License

This project is open source and available under the MIT License.

## Contributions

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests

## Support

For questions or issues, please open a GitHub issue or contact the author.

---

**Last Updated**: January 2026
**Project Status**: Active
