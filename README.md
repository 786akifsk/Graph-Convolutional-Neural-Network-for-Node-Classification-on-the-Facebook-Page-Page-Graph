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

## 🔧 **Technologies Used**

- **Python**
- **NumPy / SciPy**
- **PyTorch / PyTorch Geometric**
- **Matplotlib**
- **Jupyter Notebook**

---

## 🌟 **Learning Outcomes**

After completing this project, you will understand:

- How GCNs work  
- How to prepare real-world graph datasets  
- How node classification works  
- How graph structure improves model performance  

---

## 📌 **Future Improvements**

- Add Graph Attention Network (GAT)  
- Implement GraphSAGE  
- Perform hyperparameter tuning  
- Visualize embeddings (t-SNE / PCA)

---

## 🤝 **Contributing**

Feel free to submit issues or pull requests to improve the project.

