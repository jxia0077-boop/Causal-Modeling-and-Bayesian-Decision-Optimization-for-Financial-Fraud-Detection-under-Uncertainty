import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import pandas as pd

def build_fraud_detection_model():
    """
    Constructs a Causal Bayesian Network for financial fraud detection.
    
    This model captures the probabilistic dependencies between user behavior 
    (Travel, Device Ownership) and transaction anomalies (Foreign Purchase, Internet Purchase)
    to infer the posterior probability of Fraud.
    """
    # 1. Network Architecture Initialization
    # Defining the Directed Acyclic Graph (DAG) based on causal assumptions
    model = BayesianNetwork([
        ('Trav', 'Fraud'),  # Causal link: Travelling increases fraud risk
        ('Trav', 'FP'),     # Causal link: Travelling leads to foreign purchases
        ('Fraud', 'FP'),    # Indicator: Fraud often manifests as foreign purchases
        ('OC', 'IP'),       # Causal link: Computer ownership affects internet usage patterns
        ('Fraud', 'IP')     # Indicator: Fraud often manifests as internet purchases
    ])

    # 2. Parameterization: Conditional Probability Distributions (CPDs)
    # State mapping: 0 = False, 1 = True

    # [Node: Travelling]
    # Prior probability of a customer travelling abroad.
    cpd_trav = TabularCPD(variable='Trav', variable_card=2, values=[[0.95], [0.05]], 
                          state_names={'Trav': ['False', 'True']})

    # [Node: Owns Computer]
    # Demographic prior: Probability of a customer owning a computer.
    cpd_oc = TabularCPD(variable='OC', variable_card=2, values=[[0.30], [0.70]],
                        state_names={'OC': ['False', 'True']})

    # [Node: Fraud]
    # Conditional probability of Fraud given Travel status.
    # Risk is higher when the customer is travelling (0.01) vs not travelling (0.004).
    cpd_fraud = TabularCPD(variable='Fraud', variable_card=2, 
                           values=[[0.996, 0.99],   # P(Not Fraud)
                                   [0.004, 0.01]],  # P(Fraud)
                           evidence=['Trav'], evidence_card=[2],
                           state_names={'Fraud': ['False', 'True'], 'Trav': ['False', 'True']})

    # [Node: Foreign Purchase (FP)]
    # Modeling the likelihood of a foreign purchase.
    # Logic: 
    # - If Travelling: High probability of foreign purchase (0.90) regardless of fraud.
    # - If Not Travelling: Low probability (0.01), but spikes if Fraud is present (0.10).
    cpd_fp = TabularCPD(variable='FP', variable_card=2,
                        values=[
                            # Trav=False (F=F, F=T) | Trav=True (F=F, F=T)
                            [0.99, 0.90, 0.10, 0.10], # FP = False
                            [0.01, 0.10, 0.90, 0.90]  # FP = True
                        ],
                        evidence=['Trav', 'Fraud'], evidence_card=[2, 2],
                        state_names={'FP': ['False', 'True'], 
                                     'Trav': ['False', 'True'], 
                                     'Fraud': ['False', 'True']})

    # [Node: Internet Purchase (IP)]
    # Modeling online transaction patterns based on device ownership and fraud status.
    # Note: These parameters define the sensitivity of the 'IP' indicator.
    # - Users without computers rarely buy online unless it is fraud.
    # - Fraudulent transactions have a higher likelihood of being online.
    
    # Calibrated probabilities based on domain constraints:
    # P(IP=T | OC=T, Fraud=F) = 0.01 (Baseline online usage)
    # P(IP=T | OC=T, Fraud=T) = 0.02 (Elevated risk)
    # P(IP=T | OC=F, Fraud=F) = 0.001 (Rare event)
    # P(IP=T | OC=F, Fraud=T) = 0.011 (Elevated risk)
    
    cpd_ip = TabularCPD(variable='IP', variable_card=2,
                        values=[
                            # OC=False (F=F, F=T) | OC=True (F=F, F=T)
                            [0.999, 0.989, 0.99, 0.98], # IP = False
                            [0.001, 0.011, 0.01, 0.02]  # IP = True
                        ],
                        evidence=['OC', 'Fraud'], evidence_card=[2, 2],
                        state_names={'IP': ['False', 'True'], 
                                     'OC': ['False', 'True'], 
                                     'Fraud': ['False', 'True']})

    # 3. Model Validation
    model.add_cpds(cpd_trav, cpd_oc, cpd_fraud, cpd_fp, cpd_ip)
    assert model.check_model(), "Model structure or CPDs are inconsistent."
    return model

def calculate_expected_utility(prob_fraud, cost_fraud=-1000, cost_block=-10, cost_ok=0):
    """
    Implements Bayesian Decision Theory to determine the optimal action.
    
    Calculates the Expected Utility (EU) for 'Blocking' vs 'Allowing' a transaction
    based on the posterior probability of fraud and the associated cost matrix.
    
    Parameters:
        prob_fraud (float): Posterior probability P(Fraud | Evidence)
        cost_fraud (int): Financial loss if fraud is missed (False Negative)
        cost_block (int): Administrative cost/User friction if blocked (True Positive / False Positive)
    
    Returns:
        tuple: (EU_Allow, EU_Block, Optimal_Action)
    """
    # Utility of Allowing: Risk-weighted cost of potential fraud
    eu_allow = prob_fraud * cost_fraud + (1 - prob_fraud) * cost_ok
    
    # Utility of Blocking: Fixed cost of intervention
    eu_block = cost_block
    
    # Decision Rule: Maximize Expected Utility
    decision = "BLOCK" if eu_block > eu_allow else "ALLOW"
    return eu_allow, eu_block, decision

if __name__ == "__main__":
    # 1. Model Initialization
    model = build_fraud_detection_model()
    infer = VariableElimination(model)
    
    print("=== Bayesian Network Architecture ===")
    print(f"Nodes: {model.nodes()}")
    print(f"Dependencies: {model.edges()}")
    print("===================================\n")

    # 2. Diagnostic Inference Case Studies
    
    # Case A: Baseline Risk (Prior)
    q_prior = infer.query(variables=['Fraud'])
    print(f"Baseline Fraud Risk (Prior):\n{q_prior}")

    # Case B: Evidence Propagation - Foreign Purchase
    # Observing 'FP=True' updates the belief about Fraud via Bayes' Theorem.
    q_fp = infer.query(variables=['Fraud'], evidence={'FP': 'True'})
    prob_fraud_given_fp = q_fp.values[1]
    print(f"\n[Inference] Posterior Fraud Probability given 'Foreign Purchase': {prob_fraud_given_fp:.4f}")

    # Case C: Evidence Propagation - Combined Indicators
    # Observing both 'FP=True' and 'IP=True'.
    q_fp_ip = infer.query(variables=['Fraud'], evidence={'FP': 'True', 'IP': 'True'})
    prob_fraud_given_combined = q_fp_ip.values[1]
    print(f"\n[Inference] Posterior Fraud Probability given 'Foreign & Internet Purchase': {prob_fraud_given_combined:.4f}")

    # 3. Decision Analysis (Utility Maximization)
    print("\n=== Bayesian Decision Analysis ===")
    print("Evaluating optimal intervention strategy under uncertainty...")
    
    # Scenario: Decision based on observing a Foreign Purchase
    eu_allow, eu_block, action = calculate_expected_utility(prob_fraud_given_fp)
    
    print(f"  > Evidence: Foreign Purchase Detected")
    print(f"  > P(Fraud | Evidence): {prob_fraud_given_fp:.4f}")
    print(f"  > Expected Utility (Allow): {eu_allow:.2f}")
    print(f"  > Expected Utility (Block): {eu_block:.2f}")
    print(f"  > OPTIMAL ACTION: {action}")