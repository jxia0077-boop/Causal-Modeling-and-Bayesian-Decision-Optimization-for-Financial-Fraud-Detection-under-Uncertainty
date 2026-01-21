# Causal Modeling and Bayesian Decision Optimization for Financial Fraud Detection under Uncertainty

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Library](https://img.shields.io/badge/pgmpy-0.1.23-orange)](https://pgmpy.org/)
[![Focus](https://img.shields.io/badge/Focus-Causal_Inference-green)]()
[![Status](https://img.shields.io/badge/Status-Research_Prototype-success)]()

---

## 📖 Abstract

Financial fraud detection is inherently challenging due to extreme class imbalance and the complex, probabilistic nature of transaction anomalies. Traditional rule-based systems often fail to capture the conditional dependencies between user behavior, context, and fraud indicators.

This project implements a **Causal Bayesian Network (CBN)** to explicitly model the probabilistic and causal relationships between latent variables (e.g., *Travel Status*, *Device Ownership*) and observed evidence (e.g., *Foreign Purchase*, *Internet Purchase*).  
Furthermore, the framework is extended into a **Bayesian Decision Network (BDN)**, applying **Maximum Expected Utility (MEU)** to optimize intervention strategies (*Block* vs. *Allow*) by explicitly quantifying the asymmetric costs of misclassification (False Positives vs. False Negatives).

---

## 🧠 Methodology & Network Topology

### 1. Causal Structure Construction

A **Directed Acyclic Graph (DAG)** is constructed using domain knowledge to enforce causal directionality and interpretability.

#### Causal Assumptions

- **Latent Causes**
  - `Trav` — User is traveling
  - `OC` — User owns a computer
- **Target Variable**
  - `Fraud` — Transaction is fraudulent (conditionally dependent on `Trav`)
- **Observed Effects**
  - `FP` — Foreign Purchase
  - `IP` — Internet Purchase  
  These variables are effects influenced by both user context and fraud status.

```mermaid
graph TD;
    Trav(Travel Status) --> Fraud(Fraud);
    Trav --> FP(Foreign Purchase);
    Fraud --> FP;
    OC(Owns Computer) --> IP(Internet Purchase);
    Fraud --> IP;

    style Fraud fill:#f9f,stroke:#333,stroke-width:2px;
    style FP fill:#bbf,stroke:#333,stroke-width:1px;
    style IP fill:#bbf,stroke:#333,stroke-width:1px;
## 2. Probabilistic Inference

Exact probabilistic inference is performed using **Variable Elimination**. As evidence arrives, the belief over fraud is updated dynamically:

\[
P(Fraud \mid E) = \frac{P(E \mid Fraud)\,P(Fraud)}{P(E)}
\]

where the evidence set is defined as:

\[
E = \{\text{ForeignPurchase} = \text{True},\ \text{InternetPurchase} = \text{True},\ \ldots\}
\]

This mechanism enables **real-time belief revision under uncertainty**, allowing the model to continuously update fraud risk as new signals are observed.

---

## ⚖️ Bayesian Decision Optimization

Beyond probabilistic prediction, the model incorporates **decision-theoretic reasoning** to determine optimal actions under uncertainty.

### Utility / Cost Matrix (Loss Function)

| Action | Fraud (Actual) | Legitimate (Actual) |
|------|---------------|---------------------|
| **Allow** | -1000 (Financial Loss) | 0 |
| **Block** | -10 (Admin Cost) | -10 (User Friction) |

---

### Maximum Expected Utility (MEU)

For a given action \( a \), the expected utility is defined as:

\[
EU(a \mid E) =
\sum_{s \in \{\text{Fraud},\ \neg\text{Fraud}\}}
P(s \mid E) \cdot U(a, s)
\]

The optimal decision is selected by maximizing expected utility:

\[
a^* = \arg\max_a EU(a \mid E)
\]

This formalism explicitly captures the trade-off between **fraud loss prevention** and **customer experience friction**.

---

## 💻 Tech Stack

- **Language**: Python 3.x  
- **Core Library**: `pgmpy` (Probabilistic Graphical Models for Python)  
- **Inference Algorithm**: Exact Inference (Variable Elimination)  
- **Visualization**:
  - Netica (initial prototyping)
  - Mermaid (documentation)

---

## 📂 Repository Structure

```text
├── fraud_detection_bn.py   # Main script: network construction & decision logic
├── A3_Q1a_BN.neta          # Netica binary file (prototype)
├── A3_Q2_BN.neta           # Netica binary file (extended model)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

