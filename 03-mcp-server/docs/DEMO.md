# Demo: a real session

The installed server over stdio, on StudyRAG's **public sample corpus** (Wikipedia articles). Tool results are real; the draft answer was written by Claude from the returned passages, with one invented sentence added on purpose. Re-create with `python scripts/demo.py`.

**User:** How are affiliates paid in affiliate marketing?

**→ `search_notes`**: found = True, best relevance = 1.0

- **[1] Affiliate_marketing — Section 7: Performance/affiliate marketing** (relevance 1.0): Affiliate marketing is also called "performance marketing", in reference to how sales employees are typically being compensated. Such employees are typically paid a commission for each sale they close, and sometimes are…
- **[2] Affiliate_marketing — Section 1: Affiliate marketing** (relevance 0.999): Affiliate marketing is a marketing arrangement in which affiliates receive a commission for each visit, signup or sale they generate for a merchant. This arrangement allows businesses to outsource part of the sales proce…
- **[3] Affiliate_marketing — Section 5: Predominant compensation methods** (relevance 0.997): Eighty percent of affiliate programs today use revenue sharing or pay per sale (PPS) as a compensation method, nineteen percent use cost per action (CPA), and the remaining programs use other methods such as cost per cli…

**Draft answer:**

> Affiliates receive a commission for each visit, signup or sale they generate for a merchant [2]. The commission is usually a percentage of the price of the product, but it can also be a flat rate per referral [2]. Eighty percent of affiliate programs use revenue sharing or pay per sale, and nineteen percent use cost per action [3]. Most affiliates are also paid a fixed monthly salary by the merchant [1].

**→ `check_answer`**: supported ratio = 0.75

- ✅ (0.94) Affiliates receive a commission for each visit, signup or sale they generate for a merchant.
- ✅ (1.00) The commission is usually a percentage of the price of the product, but it can also be a flat rate per referral.
- ✅ (0.85) Eighty percent of affiliate programs use revenue sharing or pay per sale, and nineteen percent use cost per action.
- ❌ (0.35) Most affiliates are also paid a fixed monthly salary by the merchant.

**Final answer** (the unsupported sentence removed):

> Affiliates receive a commission for each visit, signup or sale they generate for a merchant [2]. The commission is usually a percentage of the price of the product, but it can also be a flat rate per referral [2]. Eighty percent of affiliate programs use revenue sharing or pay per sale, and nineteen percent use cost per action [3].

**User:** What is the boiling point of mercury?

**→ `search_notes`**: found = False, best relevance = 0.0: *"Nothing relevant enough was found (best relevance below the calibrated threshold 0.2). Say the notes don't cover this rather than answering from general knowledge."*
