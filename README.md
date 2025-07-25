# Predictive-Analytics-in-Sports

## Project Overview
Cricket isn't just a game—it's a data-rich domain of strategy and performance. Inspired by personal enthusiasm for IPL and guided by practical constraints, this project applies deep learning to analyze and predict match outcomes using structured match metadata. The focus is on creating an efficient pipeline that extracts meaningful insights while remaining computationally feasible.

## Key Features
- Structured dataset including teams, toss decisions, venues, and results
- Data preprocessing using Pandas: cleaning, encoding, and standardization
- Model development with PyTorch: fully connected neural network with ReLU, dropout, and Adam optimizer
- Evaluation metrics: accuracy, cross-entropy loss, confusion matrix, and AUC-ROC
- Exploratory analysis visualizations using Matplotlib and Seaborn
- Results highlight predictive fairness and team-specific learning trends

## Tech Stack
- **Python 3.10+**
- **PyTorch**
- **Scikit-learn**
- **Pandas**
- **Matplotlib / Seaborn**
- **Jupyter Notebook / Google Colab**

## Dataset
Sourced from [Kaggle IPL Dataset](https://www.kaggle.com/datasets/rajsengo/indian-premier-league-ipl-all-seasons) and validated with [IPL T20 Official Website](https://www.iplt20.com/). Contains match-level metadata across multiple seasons including:

- Team configurations and scores
- Toss outcomes and venue details
- Player of the match and win margins

## Methodology
- Data cleaning and transformation
- Feature encoding and scaling
- Model training loop using `TensorDataset` and `DataLoader`
- Performance tracking over epochs
- Visualization of training dynamics and prediction confidence

## Learning Outcomes
- Built an end-to-end deep learning pipeline with PyTorch
- Compared classical machine learning to neural networks
- Applied multi-class classification metrics for sports analytics
- Gained hands-on experience in tuning and evaluating models

## Future Enhancements
- Hyperparameter optimization using GridSearch/Bayesian tuning
- Time-series modeling via LSTM/Transformer
- Real-time match prediction and player-level integration
- Expansion to other cricket leagues (BBL, CPL)

## Real-World Applications
- Fantasy league recommendations
- Team strategy planning dashboards
- Sports broadcasting visual analytics
- Fan engagement tools and win-prediction apps

## References
Please refer to the full `Project Work.docx` for detailed citations including:
- Python for Data Analysis (McKinney, 2017)
- Hands-On ML with Scikit-Learn & PyTorch (Geron, Raschka)
- PyTorch Documentation & NeurIPS Proceedings
- Kaggle IPL Dataset and IPLT20.com

## Contact
**Author**: Sai Mani Ritish Upadhyayula  
[saimaniritish1942@gmail.com]
