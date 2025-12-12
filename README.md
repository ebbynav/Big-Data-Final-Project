# Stock Market Trend Prediction and Strategy Analysis

**FA 25 Big Data (CS-GY 6513), Section E**  
**Fall 2025**

### Authors

- Aniruthan Rajendran (ar9660)
- Abhinav Sivakumar (as21153)
- Pranav Vaithiya Subramani (pv2369)

---

## Project Overview

This project explores two complementary approaches to stock market analysis and prediction for S&P 500 stocks:

1. **Traditional Quantitative Finance**: Factor-based portfolio strategies using the Fama-French model with PySpark for big data processing
2. **Deep Learning**: Graph Convolutional Networks (GCN) for stock trend prediction, modeling stocks as nodes in a correlation-based graph

The project demonstrates how big data technologies (Apache Spark) and modern deep learning (PyTorch Geometric) can be applied to financial data analysis at scale.

---

## Project Structure

```
Big Data Final/
├── Fama_French_Strategies.ipynb    # Factor-based portfolio strategies with PySpark
├── GCN_Stock_Prediction.ipynb      # Graph Neural Network stock prediction
├── README.md                       # This documentation file
└── data/                           # Generated data directory (created on first run)
    |── prices.csv                  # Raw stock price data (~158 MB)
    |── prices.parquet/             # Spark-optimized format
    |── fama_french_results.png     # Performance visualization
    |── strategy_comparison.png     # Strategy comparison charts
    ├── gcn_data.csv                # Ticker and preprocessed data for GCN
```

---

## Notebook 1: Fama_French_Strategies.ipynb

### Overview

This notebook implements traditional quantitative finance approaches using the Fama-French Cross-Sectional Factor Model for S&P 500 stock analysis and portfolio strategy development. It leverages **Apache Spark** for distributed data processing, enabling scalable analysis of over 1.2 million rows of stock data.

### Technical Stack

- **Apache PySpark & Spark SQL**: Distributed data processing with 16-partition parallelism
- **Arrow Optimization**: Fast data transfer between Spark and Pandas
- **yfinance**: S&P 500 historical OHLCV data retrieval (10 years of data)
- **pandas-datareader**: Fama-French 3-factor data from Ken French's data library
- **statsmodels**: OLS regression for factor analysis

### Analysis Phases

| Phase                       | Description                                            | Key Operations                             |
| --------------------------- | ------------------------------------------------------ | ------------------------------------------ |
| **1. Environment Setup**    | Initialize Spark, detect GPU, configure environment    | PySpark session with 16 partitions         |
| **2. Data Collection**      | Download S&P 500 historical prices                     | 494 stocks, 2,517 trading days, 1.2M+ rows |
| **3. Factor Engineering**   | Compute momentum, volatility, liquidity, size factors  | Window functions, cross-sectional ranking  |
| **4. Strategy Backtesting** | Compare portfolio strategies against S&P 500 benchmark | Long-short signals, daily rebalancing      |

### Strategies Implemented

| Strategy                  | Description                    | Total Return | Sharpe Ratio | Max Drawdown | Win Rate |
| ------------------------- | ------------------------------ | ------------ | ------------ | ------------ | -------- |
| **S&P 500 (Benchmark)**   | Buy-and-hold index             | 58.86%       | 1.95         | -10.28%      | 56.6%    |
| **All-Stocks (Baseline)** | Equal-weight all stocks        | 41.95%       | 1.48         | -12.72%      | 55.1%    |
| **Low-Volatility**        | Bottom 50% volatility stocks   | 32.13%       | 1.42         | -11.08%      | 57.5%    |
| **Momentum + Quality**    | High momentum + low volatility | 37.58%       | 1.59         | -11.03%      | 55.1%    |

### Key Results (Test Period: 2023-2024)

- **Data Scale**: 494 S&P 500 tickers, 1,217,083 total rows
- **Training Period**: 2014-12-08 to 2022-12-30 (977,580 rows)
- **Test Period**: 2023-01-03 to 2024-12-06 (239,503 rows)
- **Best Risk-Adjusted Strategy**: Momentum + Quality (Sharpe 1.59)
- **Fama-French Alpha**: -1.24% (market-neutral strategy)

### Fama-French Regression Results

The long-short factor portfolio was regressed against Fama-French 3 factors:

- **R-squared**: 0.72%
- **Market Beta**: -0.11 (low market exposure)
- **SMB Loading**: 0.01 (size-neutral)
- **HML Loading**: -0.08 (slight growth tilt)

---

## Notebook 2: GCN_Stock_Prediction.ipynb

### Overview

This notebook implements a **Graph Convolutional Network (GCN)** for stock market trend prediction. It models the stock market as a dynamic graph where:

- **Nodes** = Individual stocks with technical indicator features
- **Edges** = Correlation-based connections between stocks (threshold: 0.7)
- **Task** = Node-level binary classification (predict next-day up/down movement)

### Technical Stack

- **PyTorch**: Deep learning framework for model training
- **PyTorch Geometric**: Specialized library for graph neural networks
- **CUDA/GPU**: Hardware acceleration for neural network training
- **NetworkX**: Graph construction and visualization

### GCN Architecture

```
Input Layer (12 features per stock)
    ↓
GCN Layer 1 (12 → 64, ReLU, Dropout 0.4)
    ↓
GCN Layer 2 (64 → 64, ReLU, Dropout 0.4)
    ↓
GCN Layer 3 (64 → 2, Softmax)
    ↓
Output: Binary Classification (Up/Down)
```

### Model Configuration

| Parameter             | Value  | Description                    |
| --------------------- | ------ | ------------------------------ |
| Input Features        | 12     | Technical indicators per stock |
| Hidden Dimension      | 64     | GCN hidden layer size          |
| Output Classes        | 2      | Up/Down prediction             |
| Dropout Rate          | 0.4    | Regularization                 |
| Learning Rate         | 0.0005 | Adam optimizer                 |
| Weight Decay          | 1e-3   | L2 regularization              |
| Correlation Threshold | 0.7    | Edge creation threshold        |

### Node Features (12 per stock)

| Feature                         | Description               |
| ------------------------------- | ------------------------- |
| ret_1d, ret_3d, ret_5d, ret_10d | Multi-horizon returns     |
| ma5, ma10, ma20                 | Moving average ratios     |
| volatility_10d                  | Short-term volatility     |
| volume_ratio                    | Volume vs 20-day average  |
| ma_cross                        | MA5/MA20 crossover signal |
| rsi14                           | Relative Strength Index   |
| macd                            | MACD indicator            |

### Graph Construction

- **Dynamic Graphs**: Correlation matrices computed over rolling windows
- **Edge Selection**: Only pairs with correlation > 0.7 form edges
- **Temporal Modeling**: Separate graph snapshots for each time window

---

## Prerequisites & Installation

### System Requirements

- **Python**: 3.8 or higher (tested with Python 3.12)
- **RAM**: 8GB minimum (16GB recommended for Spark operations)
- **Disk Space**: ~500MB for data and models
- **Internet**: Required for initial data download
- **GPU**: Optional but recommended for GCN notebook (CUDA-compatible NVIDIA GPU)

### Required Python Packages

```bash
# Core data processing
pip install pyspark pandas numpy

# Financial data
pip install yfinance pandas-datareader

# Visualization
pip install matplotlib seaborn

# Statistical modeling
pip install statsmodels scikit-learn

# Deep learning (for GCN notebook)
pip install torch torchvision
pip install torch-geometric
```

### Quick Installation (All Packages)

```bash
pip install pyspark pandas numpy yfinance pandas-datareader matplotlib seaborn statsmodels scikit-learn torch torchvision torch-geometric
```

---

## How to Run the Notebooks

### Option 1: Running in VS Code (Recommended)

1. **Open the Project Folder**

   ```
   File → Open Folder → Select "Big Data Final"
   ```

2. **Install Python Extension** (if not already installed)

   - Search for "Python" in Extensions (Ctrl+Shift+X)
   - Install Microsoft's Python extension

3. **Select Python Kernel**

   - Open any `.ipynb` file
   - Click "Select Kernel" in the top-right
   - Choose your Python 3.8+ interpreter

4. **Run All Cells**
   - Click "Run All" button in the notebook toolbar
   - Or use `Ctrl+Alt+Enter` to run all cells

### Option 2: Running in Jupyter Lab

```bash
# Navigate to the project directory
cd "Big Data Final"

# Start Jupyter Lab
jupyter lab

# Open the notebook files from the file browser
```

### Option 3: Command Line Execution

```bash
# Convert notebooks to Python scripts and run
jupyter nbconvert --to script Fama_French_Strategies.ipynb
python Fama_French_Strategies.py
```

---

## Troubleshooting

### Common Issues

**Issue**: PyTorch DLL error on Windows

```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed
```

**Solution**: Reinstall PyTorch with the correct CUDA version, or the CPU-only version:

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Issue**: `ModuleNotFoundError: No module named 'torch_geometric'`
**Solution**: Install PyTorch Geometric:

```bash
pip install torch-geometric
```

**Issue**: Spark session fails to start
**Solution**: Ensure Java is installed and JAVA_HOME is set:

```bash
# Check Java installation
java -version
```

**Issue**: yfinance data download fails for some tickers
**Solution**: This is expected for delisted stocks. The notebook handles this gracefully.

---

## References

1. Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. _Journal of Financial Economics_.
2. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. _ICLR_.
3. Chen, X., et al. (2022). Stock Price Prediction Using Graph Neural Networks.

---

## License

**Academic Use Only** - This project was developed as part of the CS-GY 6513 Big Data course at NYU Tandon School of Engineering.
