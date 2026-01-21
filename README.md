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
--
## 📊 Experimental Results

### Case Study: *The Foreign Purchase Scenario*

When a transaction is flagged as a **Foreign Purchase** (`FP = True`), the model updates its belief state as follows:

#### 🔹 Prior Belief
- **Prior Probability**:  
  \[
  P(\text{Fraud}) \approx 0.42\%
  \]

#### 🔹 Posterior Belief
- **Posterior Probability**:  
  \[
  P(\text{Fraud} \mid FP = \text{True})
  \]
  increases significantly according to the Conditional Probability Tables (CPTs).

#### 🔹 Decision Analysis

- **Expected Utility of Allowing the Transaction**:
  \[
  EU(\text{Allow}) = P(\text{Fraud} \mid FP) \times (-1000) 
  + P(\neg \text{Fraud} \mid FP) \times 0
  \]

- **Expected Utility of Blocking the Transaction**:
  \[
  EU(\text{Block}) = -10
  \]

#### 🔹 Decision Outcome
- When the estimated fraud risk exceeds the **break-even probability threshold**, the system **automatically switches** its decision from **ALLOW** to **BLOCK**.

✔️ **Final Decision**: **BLOCK**

