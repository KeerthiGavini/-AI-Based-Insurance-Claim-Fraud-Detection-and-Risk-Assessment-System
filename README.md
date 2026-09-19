# AI-Based Insurance Claim Fraud Detection and Risk Assessment System

##  Project Overview

The **AI-Based Insurance Claim Fraud Detection and Risk Assessment System** is a web-based platform designed to support insurance companies in identifying potentially fraudulent insurance claims and prioritizing them for further investigation.

The system uses **Machine Learning and Deep Learning** techniques to analyze insurance claim and policy-related information. Multiple models, including **Logistic Regression, Random Forest, XGBoost, and Artificial Neural Network (ANN)**, are trained and evaluated to identify an effective approach for fraud detection.

For each submitted claim, the system generates a **fraud probability**, converts it into a **risk score**, and classifies the claim into **Low, Medium, High, or Critical** risk levels.

The system also integrates **Explainable Artificial Intelligence (XAI)** using **SHAP** to identify the important features contributing to an individual prediction. High-risk claims are prioritized for investigation, where authorized investigators can review claim details, examine AI predictions and explanations, add investigation notes, update investigation status, and record the final decision.

The platform combines **claim management, AI-based fraud detection, risk assessment, explainable AI, investigation management, dashboards, analytics, notifications, reporting, and audit logging** into a unified application.

---

##  Problem Statement

Insurance fraud is a major challenge for insurance companies because fraudulent claims can result in financial losses and increase the time and effort required for claim verification.

Manual claim verification can make it difficult to efficiently identify suspicious claims, particularly when a large number of claims need to be processed. There is a need for an intelligent system that can analyze claim information, identify potentially fraudulent claims, assess their risk level, and provide useful explanations to support investigation.

The proposed system addresses these requirements by combining **Machine Learning, Deep Learning, Risk Assessment, Explainable AI, and Investigation Management** into a single web-based platform.

---

##  Proposed Solution

The proposed system provides a centralized platform for managing insurance claims and supporting fraud investigation.

The system processes claim and policy-related information through data preprocessing, missing-value handling, feature engineering, categorical encoding, feature scaling, and class-imbalance handling.

Multiple Machine Learning and Deep Learning models are trained and evaluated. The selected model generates a fraud probability for a new claim. The probability is then converted into a risk score and risk level.

SHAP-based Explainable AI provides information about the important features contributing to the prediction. Claims with higher risk levels can be prioritized for investigation.

Authorized investigators can review the claim, examine the AI prediction and explanation, add investigation notes, update the investigation status, and record the final decision.

---

## Objectives

The main objectives of the project are:

- To develop an AI-based insurance claim fraud detection platform.
- To analyze insurance claim and policy-related information.
- To detect potentially fraudulent insurance claims.
- To generate fraud probability for submitted claims.
- To calculate a risk score for each claim.
- To classify claims into Low, Medium, High, and Critical risk levels.
- To compare Machine Learning and Deep Learning models.
- To provide prediction explanations using SHAP.
- To prioritize high-risk claims for further investigation.
- To provide investigators with claim review and investigation management functionality.
- To provide dashboards and analytics for monitoring claims.
- To implement secure authentication and role-based access control.
- To maintain audit records of important system activities.

---

#  Key Features

## User Authentication and Authorization

The platform provides secure user authentication and authorization. Different users can access different system functions based on their assigned roles.

The system can support roles such as:

- Admin
- Investigator
- Analyst
- Other authorized users

Role-based access control ensures that users can access only the functions permitted to their roles.

---

## Customer Management

The customer management module maintains information related to insurance customers.

Users can:

- Add customer information
- View customer details
- Update customer information
- Search customers
- View associated policies and claims

---

##  Policy Management

The policy management module stores and manages insurance policy information.

The system maintains relevant information such as:

- Policy details
- Policy type
- Policy dates
- Policy status
- Customer information
- Claim association

---

##  Insurance Claim Management

The claim management module allows authorized users to manage insurance claims.

Users can:

- Submit claims
- View claims
- Update claim information
- Search claims
- View claim status
- View fraud prediction results
- View risk information

---

##  AI-Based Fraud Detection

The system uses Machine Learning and Deep Learning models to analyze insurance claim information and identify potentially fraudulent claims.

The implemented models include:

- Logistic Regression
- Random Forest
- XGBoost
- Artificial Neural Network (ANN)

The models are evaluated using suitable performance metrics before selecting the model for the final prediction workflow.

---

## Fraud Probability Prediction

For every submitted claim, the selected AI model generates a fraud probability.

The probability represents the model's estimated likelihood that the claim belongs to the fraudulent class.

The probability is then used as an input for the risk assessment process.

---

##  Risk Assessment

The system converts the predicted fraud probability into a risk score.

The claim is then classified into:

- Low
- Medium
- High
- Critical

This allows claims to be prioritized according to their predicted risk.

---

##  Explainable AI using SHAP

The system uses **SHAP (SHapley Additive exPlanations)** to explain individual predictions.

SHAP identifies the important features that contributed to the prediction and helps authorized users understand the factors associated with a particular fraud prediction.

The explanation can show:

- Important contributing features
- Features increasing fraud probability
- Features decreasing fraud probability
- Contribution of individual features

---

##  High-Risk Claim Prioritization

Claims classified as High or Critical risk can be prioritized for further investigation.

This allows investigators to focus their attention on claims that require additional review.

---

##  Investigation Management

The investigation module provides a structured process for reviewing suspicious claims.

Authorized investigators can:

- View high-risk claims
- Review claim details
- View fraud probability
- View risk score
- View risk level
- View SHAP explanations
- Add investigation notes
- Update investigation status
- Record investigation decisions

---

##  Dashboard and Analytics

The dashboard provides a centralized view of important system information.

It can display:

- Total customers
- Total policies
- Total claims
- Fraud-related statistics
- Risk-level distribution
- Investigation statistics
- Recent activities
- Model performance information
- Analytics charts

---

##  Notifications

The notification module provides updates about important system activities.

Notifications can include:

- New claim submission
- High-risk claim identification
- Investigation assignment
- Investigation status changes
- Comments or updates
- Important system activities

---

##  Report Generation

The system provides report-generation functionality for relevant information such as:

- Claim reports
- Fraud detection reports
- Risk assessment reports
- Investigation reports
- Model evaluation reports
- Analytics reports

---

##  Audit Logging

The audit logging module records important activities performed within the system.

Audit information can include:

- User activity
- Login activity
- Claim updates
- Policy updates
- Investigation updates
- Administrative actions

---

# System Architecture

The overall system architecture is represented below:

```text
                         ┌─────────────────────┐
                         │        Users        │
                         │ Admin / Investigator│
                         │ Analyst / Others    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Frontend        │
                         │      React.js       │
                         └──────────┬──────────┘
                                    │
                              REST API
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Backend        │
                         │       FastAPI       │
                         │ Authentication      │
                         │ Claim Management    │
                         │ Investigation       │
                         │ AI/ML Services      │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
                  ▼                 ▼                 ▼
          ┌──────────────┐  ┌──────────────┐  ┌───────────────┐
          │   Database   │  │ AI/ML Engine │  │  SHAP Engine  │
          │              │  │              │  │               │
          │ Users        │  │ Preprocessing│  │ Explanations  │
          │ Customers    │  │ ML Models    │  │ Feature       │
          │ Policies     │  │ DL Model     │  │ Contributions │
          │ Claims       │  │ Evaluation   │  │               │
          │ Investigation│  │ Prediction   │  │               │
          └──────────────┘  └──────┬───────┘  └───────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │ Fraud Probability   │
                         │ Risk Score           │
                         │ Risk Level           │
                         │ SHAP Explanation     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Investigation Module│
                         │ Review / Notes      │
                         │ Status / Decision   │
                         └─────────────────────┘
