# 📘 An Optuna-Tuned Graph Convolutional Neural Network for Facebook Page–Page Node Classification
### **Node Classification on the Facebook Page–Page Graph**

This project applies a **Graph Convolutional Neural Network (GCN)** to perform **node classification** on the **Facebook Page–Page graph**.

Each node represents a Facebook page, and edges represent relationships between pages.  
The goal is to predict the **category of each page** using node features and graph structure.

---

## 📂 Project Structure

```
Graph-Convolutional-Neural-Network-for-Node-Classification-on-the-Facebook-Page-Page-Graph/
│
├── ORIGINAL DATASETS/          # Graph dataset files
│   ├── edges.csv               # Page–Page graph edges (src, dst)
│   ├── features.npy            # Node features matrix
│   └── labels.npy              # Node labels (categories)
│
├── NOTEBOOK FILES/             # Jupyter notebooks for training & analysis
│   ├── Data_Preprocessing.ipynb
│   ├── GCN_Training.ipynb
│   └── Evaluation.ipynb
│
└── README.md                   # Project overview
```

---

## 🎯 **Project Objective**

The objective of this project is to:

- Build a **GCN model** capable of learning from graph-structured data  
- Use the Page–Page graph to classify Facebook pages into categories  
- Understand how **node connectivity** influences predictions  
- Analyze performance through evaluation metrics and visualizations  

---

## 🧠 **What Is a GCN?**

A **Graph Convolutional Network (GCN)** is a neural network designed to operate directly on **graph data**.

Key ideas:

- Nodes aggregate information from their neighbors  
- Graph structure contributes to learning  
- Useful for social networks, citation networks, chemistry, etc.  

GCNs help answer:  
**"Can we classify a node using its features AND graph connections?"**

---

## ⚙️ **Project Workflow**

### **1️⃣ Data Preprocessing**
- Load feature matrix, label vector, and edge list  
- Construct adjacency matrix  
- Normalize graph  
- Prepare data for GCN input  

### **2️⃣ Model Architecture**
- Two-layer Graph Convolutional Network  
- ReLU activation  
- Dropout regularization  

### **3️⃣ Training Process**
- Node splits: Train / Validation / Test  
- Cross-entropy optimization  
- Accuracy tracking over epochs  

### **4️⃣ Evaluation**
- Final **test accuracy**  
- Class-wise performance (optional)  
- Visualization of predictions (optional)

---

## 🚀 **How to Run the Project**

1. Place your dataset in the folder: **ORIGINAL DATASETS/**
2. Open notebooks in this order:
   - **Data_Preprocessing.ipynb**
   - **GCN_Training.ipynb**
   - **Evaluation.ipynb**
3. Run each notebook sequentially.

---

## 📊 **Dataset Used**

The Facebook Page–Page graph includes:

- **Nodes** → Facebook pages  
- **Edges** → Connections between pages  
- **Features** → Attributes of each page  
- **Labels** → Categories of pages  

This is a commonly used benchmark dataset for graph machine learning.

---

## 🛠️ Technologies Used

|         Category        |             Tools            |
| ----------------------- |:----------------------------:|
| Programming Languages   | Python 3.12, PyTorch         |
| Graph Learning          | PyTorch Geometric            |
| Optimization            | Optuna                       |
| Visualization           | Matplotlib, Seaborn          |
| Preprocessing           | NumPy, Pandas, Scikit-learn  |
| Environment             | Google Colab, VS Code , Github       |

---

## 🌟 **Learning Outcomes**

After completing this project, you will understand:

- How GCNs work  
- How to prepare real-world graph datasets  
- How node classification works  
- How graph structure improves model performance  

---

## 🚀 Applications
* Social network analysis
* Automated content classification
* Influence detection
* Community structure analysis
* Recommendation systems
* Political or organizational group detection
  
---

## 📌 **Future Improvements**

* Evaluate the GCN model on other social network datasets to test performance across different graph structures.
* Compare results between datasets to check model robustness and generalizability.
* Analyze how the model adapts to varying node features and connectivity patterns.
* Improve architecture with advanced GNN variants such as GraphSAGE, GAT, or GCNII.
* Explore scalability for large-scale real-world social network applications.

---

## 🏁 Conclusion
The project successfully demonstrates that a Graph Convolutional Neural Network (GCNN) can effectively classify Facebook pages by leveraging both node features and the underlying network structure. Through proper data preprocessing, graph construction, and hyperparameter tuning using Optuna, the model achieved strong and stable performance with a final test accuracy of around 73%. The evaluation metrics-including ROC–AUC, confusion matrix, and accuracy curves—show that the model generalizes well across classes and captures meaningful relationships in the graph. Overall, the work highlights the power of graph-based deep learning for real-world social network analysis and classification tasks.

---

## 👥 Contributors

| Name | GitHub | LinkedIn |
|------|--------|----------|
| Arnab Paul | [![GitHub](https://img.shields.io/badge/-GitHub-000?logo=github&logoColor=white)](https://github.com/arnabpaul873-star) | [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arnab-paul-927930253/) |
| Utkarsh Abhishek | [![GitHub](https://img.shields.io/badge/-GitHub-000?logo=github&logoColor=white)](https://github.com/TheDesperateCoder) | [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/utkarsh-a-a1194911b/) |
| Md Akif Sk | [![GitHub](https://img.shields.io/badge/-GitHub-000?logo=github&logoColor=white)](https://github.com/786akifsk) | [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/md-akif-sk-4743461b7/) |
| Deeparghya Ghosh | [![GitHub](https://img.shields.io/badge/-GitHub-000?logo=github&logoColor=white)](https://github.com/Deep8095) | [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?logo=linkedin&logoColor=white)](LINK) |

