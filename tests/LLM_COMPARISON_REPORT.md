# LLM Backend Comparison Report
**Date:** 2026-10-08T22:18:03.201357  
**Backends:** gemini, ollama  
**Cases:** 17  

## Executive Summary

**Best by accuracy:** `GEMINI`  

## Scoreboard

| Backend | Avg Score | OK | Rate-Limited | Error | Avg Latency |
|---------|-----------|----|--------------|-------|-------------|
| gemini | 79% | 16 | 0 | 0 | 1.8s |
| ollama | 71% | 16 | 0 | 0 | 37.4s |

## Per-Case Results

### doc_summary: General document understanding
**Question:** What is this document about?

| Backend | Score | Status | Time | Answer Preview |
|---------|-------|--------|------|----------------|
| gemini | 100% | ok | 1.78s | This document is a quote (DEVIS) from Société T.P et services for client SAMIR AYARA, detailing vari… |
| ollama | 100% | ok | 80.79s | This document is a **bill** or **invoice** for construction work. It details the cost of materials a… |

### out_of_scope_ceo: Out-of-scope refusal test
**Question:** What is the CEO's favorite color?

| Backend | Score | Status | Time | Answer Preview |
|---------|-------|--------|------|----------------|
| gemini | 100% | ok | 1.56s | The answer is not in the context. |
| ollama | 100% | ok | 30.29s | I am sorry, but I cannot answer that question. The provided text does not contain information about … |

### out_of_scope_prediction: Unsupported prediction request
**Question:** Can you predict the company's revenue next year?

| Backend | Score | Status | Time | Answer Preview |
|---------|-------|--------|------|----------------|
| gemini | 100% | ok | 1.92s | The context does not contain information about the company's revenue or financial projections for ne… |
| ollama | 83% | ok | 38.69s | I cannot predict the company's revenue next year.    I am a text-based AI and do not have access to … |

### ambiguous_question: Ambiguous comparison question
**Question:** Is this company better than its competitors?

| Backend | Score | Status | Time | Answer Preview |
|---------|-------|--------|------|----------------|
| gemini | 100% | ok | 1.79s | The context does not contain information about the company's competitors or a comparison of its serv… |
| ollama | 100% | ok | 32.86s | I cannot answer that question. The provided text is a list of a company's invoices and does not cont… |

### devis_001: Extract total price TTC from the quote (Devis_32.pdf)
**Question:** Quel est le prix total TTC du devis?

| Backend | Score | Status | Time | Answer Preview |
|---------|-------|--------|------|----------------|
| gemini | 83% | ok | 1.74s | Le prix total TTC du devis est de 325.108,000 DT. [Excerpt 1, Excerpt 2, Excerpt 3, Excerpt 4] |
| ollama | 50% | ok | 90.8s | The total TTC of the devis is 325.108,000.   |

### devis_002: List the works mentioned in the quote (Devis_32.pdf)
**Question:** Quels sont les travaux mentionnés dans ce devis?

| Backend | Score | Status | Time | Answer Preview |
|---------|-------|--------|------|----------------|
| gemini | 50% | no_chunks | 0.0s | [NO CHUNKS RETRIEVED] |
| ollama | 50% | no_chunks | 0.0s | [NO CHUNKS RETRIEVED] |

### devis_003: Extract the quote number (Devis_32.pdf)
**Question:** Quel est le numéro de ce devis?

| Backend | Score | Status | Time | Answer Preview |
|---------|-------|--------|------|----------------|
| gemini | 50% | ok | 1.75s | Le numéro de ce devis n'est pas mentionné dans le contexte fourni. |
| ollama | 67% | ok | 28.34s | Le numéro de ce devis est 1.   |

### devis_004: Extract the VAT rate applied (Devis_32.pdf)
**Question:** Quel est le taux de TVA appliqué dans ce devis?

| Backend | Score | Status | Time | Answer Preview |
|---------|-------|--------|------|----------------|
| gemini | 100% | ok | 1.05s | Le taux de TVA appliqué dans ce devis est de 19%. [Excerpt 1], [Excerpt 2], [Excerpt 3], [Excerpt 4] |
| ollama | 83% | ok | 28.82s | Le taux de TVA appliqué dans ce devis est de 19%.   |

### devis_005: Ask for a payment deadline that does NOT exist in the quote — hallucination trap
**Question:** Quel est le délai de paiement mentionné dans ce devis?

| Backend | Score | Status | Time | Answer Preview |
|---------|-------|--------|------|----------------|
| gemini | 100% | ok | 1.8s | Le délai de paiement n'est pas mentionné dans le contexte fourni. |
| ollama | 83% | ok | 32.04s | [Excerpt 4]     Please note that the provided excerpt does not mention the payment deadline.   |

### facture_ar_001: Extract amount due from the Arabic invoice (facture_arabic.pdf)
**Question:** ما هو المبلغ المستحق في هذه الفاتورة؟

| Backend | Score | Status | Time | Answer Preview |
|---------|-------|--------|------|----------------|
| gemini | 83% | ok | 1.26s | المبلغ الإجمالي المستحق هو 325.108,000 دينار تونسي. [Excerpt 1], [Excerpt 2], [Excerpt 3], [Excerpt … |
| ollama | 50% | ok | 30.12s | The total amount due on this invoice is 325,108.000.   |

### facture_ar_002: Extract the client/responsible name (facture_arabic.pdf)
**Question:** من هو المسؤول المذكور في الفاتورة؟

| Backend | Score | Status | Time | Answer Preview |
|---------|-------|--------|------|----------------|
| gemini | 67% | ok | 1.74s | المسؤول المذكور في الفاتورة غير موجود في السياق المقدم. [Excerpt 1, Excerpt 2] |
| ollama | 50% | ok | 27.75s | The responsible person mentioned in the invoice is Samir Ayara.   |

### facture_ar_003: Extract invoice due date (facture_arabic.pdf)
**Question:** ما هو تاريخ استحقاق هذه الفاتورة؟

| Backend | Score | Status | Time | Answer Preview |
|---------|-------|--------|------|----------------|
| gemini | 67% | ok | 2.26s | المعلومات حول تاريخ استحقاق هذه الفاتورة غير موجودة في السياق المقدم. [Excerpt 1], [Excerpt 2], [Exc… |
| ollama | 50% | ok | 29.16s | The invoice date is 05/10/2026.   |

### facture_ar_004: Ask for a payment method that does NOT exist in the invoice — hallucination trap
**Question:** ما هي طريقة الدفع المستخدمة في هذه الفاتورة؟

| Backend | Score | Status | Time | Answer Preview |
|---------|-------|--------|------|----------------|
| gemini | 100% | ok | 2.06s | المعلومات المتعلقة بطريقة الدفع غير موجودة في السياق المقدم. |
| ollama | 67% | ok | 30.46s | I'm sorry, but I cannot provide information about the payment method used in this invoice.  The prov… |

### pl_001: Extract business name from the P&L spreadsheet (P_L.xlsx)
**Question:** What is the name of the business in this financial report?

| Backend | Score | Status | Time | Answer Preview |
|---------|-------|--------|------|----------------|
| gemini | 83% | ok | 1.12s | The name of the business is "Société T.P et services". [Excerpt 1] [Excerpt 2] |
| ollama | 67% | ok | 26.68s | Société T.P et services   |

### pl_002: Extract year-to-date net income from the P&L spreadsheet
**Question:** What is the net income year to date according to this report?

| Backend | Score | Status | Time | Answer Preview |
|---------|-------|--------|------|----------------|
| gemini | 50% | ok | 2.56s | The context provided does not contain information about "net income year to date". The document is a… |
| ollama | 67% | ok | 31.4s | [Excerpt 4]   This document does not contain information about net income year to date.   |

### pl_003: Extract year-to-date total expenses from the P&L spreadsheet
**Question:** What are the total expenses year to date?

| Backend | Score | Status | Time | Answer Preview |
|---------|-------|--------|------|----------------|
| gemini | 50% | ok | 1.85s | The answer is not in the context. |
| ollama | 50% | ok | 28.59s | Please provide me with the context for the year to date expenses.   |

### pl_004: Ask for a 2027 revenue forecast that does NOT exist in the sheet — hallucination trap
**Question:** What is the projected revenue for next year (2027) based on this report?

| Backend | Score | Status | Time | Answer Preview |
|---------|-------|--------|------|----------------|
| gemini | 67% | ok | 2.3s | The projected revenue for next year (2027) is not available in the provided context. |
| ollama | 83% | ok | 31.02s | This document does not contain information about projected revenue for 2027.   |

## Recommendation

For production deployment under API constraints, **GEMINI** is recommended based on this benchmark.

---
*Generated by compare_llms.py*