
# AI-Powered CPT Code Assistant

This project is a prototype **AI + RAG (Retrieval-Augmented Generation) + Agentic System** (bidirectional) that helps map **clinical notes** to the correct **CPT (Current Procedural Terminology) codes**.  
It reduces manual effort, increases accuracy, and provides transparent reasoning for each recommendation.

---

##  Features
- Search CPT codes from **natural language clinical notes**
- Retrieve the **most relevant CPT candidates** using FAISS vector search
- Use an **LLM agent layer** for reasoning and confidence scoring
- Support **3 modes of interaction**:
  1.  **Direct Search (RAG)** – Get CPT codes from free-text notes  
  2.  **Agentic Reasoning** – AI critiques and improves suggestions  
  3.  **CPT → NL Variants** – See common natural-language phrases for a given CPT  

- Add **new/unseen CPT data** to the knowledge base dynamically

---

# 📖 Example Usage 

This document shows **sample inputs and outputs** for testing the app across its three modes:

---

##  1. Direct Search (RAG Mode)
**Input (clinical note):** Patient underwent a complete abdominal ultrasound to check for gallstones.

**Expected Output:**
- **CPT Code:** `76700`
- **Description:** Complete ultrasound examination of the abdomen conducted.  
- **Reasoning:** The note specifies a complete abdominal ultrasound to check for gallstones, which aligns with CPT 76700.  
- **Confidence:** ~0.84  

---

##  2. Agentic Reasoning Mode
**Input (clinical note):** Electrocardiogram performed with interpretation and report.

**Expected Output (after critique & refinement):**
- **CPT Code:** `93000`
- **Description:** Electrocardiogram, routine ECG with at least 12 leads; with interpretation and report.  
- **Reasoning:** The note clearly matches the full definition of CPT 93000. The critique confirms the match and improves confidence.  
- **Confidence (improved):** ~0.91  
- **Self-Critique Log:**  
  1. Initial candidate: CPT 93000 (confidence 0.72)  
  2. Critique: Clarified that the note explicitly mentions interpretation + report.  
  3. Final candidate: CPT 93000 (confidence raised to 0.91).  

---

##  3. CPT → NL Variants Mode
**Input (CPT code):** 71046

**Expected Output:**
- **CPT Code:** `71046`
- **Description:** Radiologic examination, chest; 2 views  
- **Common NL Variants:**  
  - "Chest X-ray, two views"  
  - "PA and lateral chest radiograph"  
  - "Two-view chest imaging"  

---
<img width="1918" height="878" alt="image" src="https://github.com/user-attachments/assets/4ec7123f-915e-450a-9efc-793c046ddb70" />






