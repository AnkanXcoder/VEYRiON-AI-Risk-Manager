# VEYRiON AI Risk Manager

Defense-only, cost-sensitive fraud-risk decision support for Razorpay Hackathon Track 02.

VEYRiON evaluates a transaction using an XGBoost fraud model, converts the predicted probability into an operational `APPROVE`, `REVIEW`, or `BLOCK` decision, and provides local SHAP evidence for model transparency.

> **Prototype only:** This project uses a public credit-card fraud benchmark and is not production-ready. No real payment is processed.

---

## 🚀 Live Demo

**Production deployment:**

https://veyrion-ai-risk-manager.onrender.com/

The deployed service runs the Flask API and the frontend together.

> **Note:** The free Render instance may spin down after inactivity, so the first request after inactivity can take longer to respond.

---

# 🎯 Problem

Fraud detection is not only a classification problem.

A merchant must decide what to do with a transaction when the model is uncertain.

A system that blocks too aggressively can create unnecessary customer friction, while a system that misses fraud can create merchant loss.

VEYRiON therefore separates:

- **Model prediction**
- **Risk policy**
- **Merchant action**
- **Model explanation**

The system routes transactions into:

```text
APPROVE
REVIEW
BLOCK