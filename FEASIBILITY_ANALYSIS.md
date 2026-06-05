# Chapters 5-6 Scope Broadening & Reference Strengthening: Feasibility Analysis

## Point 20: Broadening Chapters 5-6 from OST to General Stopping Time Strategy

### Current State
- **Chapter 5**: "最优停时与 Snell 包络：秘书问题" — framed as "OST doesn't apply, so we need Snell envelope"
- **Chapter 6**: "美式期权定价与 Longstaff-Schwartz 算法" — framed as "Snell envelope + LSM numerical method"
- The OST connection in these chapters is tenuous: OST mainly serves as a bridge concept, but the real contributions are optimal stopping theory and numerical methods.

### Proposed Change
Broaden titles to emphasize stopping time strategy analysis:
- Chapter 5: "最优停时策略分析：秘书问题" (Optimal Stopping Strategy Analysis: Secretary Problem)
- Chapter 6: "最优停时的数值方法：美式期权定价" (Numerical Methods for Optimal Stopping: American Option Pricing)

### Impact Analysis
- **Chapter 5 restructuring**: Introduction needs to shift from "OST doesn't work here" to "this is about choosing optimal stopping times." Section 5.2.1 (record indicator is not a martingale) still serves as motivation, but the emphasis changes. Section 5.2.4 (Why OST doesn't conflict) may be shortened or integrated into 5.2.1.
- **Chapter 6 restructuring**: Introduction shifts from Snell envelope continuation to a broader view of numerical optimal stopping. The Snell envelope remains core, but the framing emphasizes the computational challenge.
- **Chapter 1 updates**: The spectrum description in 1.1.2 (already rewritten) aligns well with this broadening.
- **Chapter 7 updates**: The comparison table and theoretical spectrum sections need minor adjustments.
- **Estimated effort**: ~2-3 hours of LaTeX editing. No new math required.

### Feasibility Verdict: **HIGH** — purely textual reframing, no new mathematical content needed.

---

## Point 22: Strengthening References

### Current State
7 references total, concentrated on core textbooks:
- Durrett (2019), Williams (1991), Longstaff & Schwartz (2001)
- Qian & Gong (2021), Ren (2022)
- Moran (1958), Asmussen & Albrecher (2010)

### Weak Areas Identified

**Chapter 1 (OST Theory)**:
- Missing: Doob's original work. Suggested: Doob, J.L. (1953). *Stochastic Processes*. Wiley.
- The indicator decomposition technique and OST proof could cite more than just Williams.

**Chapter 2 (Moran Model)**:
- Only cites Moran (1958). Missing:
  - Ewens, W.J. (2004). *Mathematical Population Genetics*. Springer. (for birth-death chain analysis)
  - The tridiagonal system solution method could cite a numerical linear algebra reference.

**Chapter 3 (Insurance Ruin)**:
- Only cites Asmussen & Albrecher (2010). Missing:
  - Gerber, H.U. (1979). *An Introduction to Mathematical Risk Theory*. Huebner Foundation.
  - Grandell, J. (1991). *Aspects of Risk Theory*. Springer.
  - The exponential martingale construction is a standard technique that could cite multiple sources.

**Chapter 4 (Gambler's Ruin)**:
- Uses Qian/Gong and Ren textbooks (Chinese language). Missing:
  - Feller, W. (1968). *An Introduction to Probability Theory and Its Applications*, Vol. 1, 3rd ed. Wiley.
  - Spitzer, F. (1964). *Principles of Random Walk*. Van Nostrand.
  - Feller's treatment of the gambler's ruin and the n^{-1/2} tail behavior is canonical.

**Chapter 5 (Secretary Problem)**:
- **No specific reference at all** aside from Williams for Snell envelope theory. This is the most severe gap. Missing:
  - Ferguson, T.S. (1989). "Who solved the secretary problem?" *Statistical Science*, 4(3), 282-296. (definitive survey)
  - Gilbert, J.P. & Mosteller, F. (1966). "Recognizing the maximum of a sequence." *JASA*, 61, 35-73.
  - Freeman, P.R. (1983). "The secretary problem and its extensions: A review." *Int. Stat. Rev.*, 51(2), 189-206.

**Chapter 6 (American Options)**:
- Only cites Longstaff & Schwartz (2001). Missing:
  - Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
  - Karatzas, I. & Shreve, S.E. (1998). *Methods of Mathematical Finance*. Springer.
  - The Black-Scholes formula could cite Black & Scholes (1973) or Merton (1973).

### Suggested Additions (~12 references)
1. Doob (1953) — OST theory foundation
2. Ewens (2004) — population genetics
3. Gerber (1979) — insurance ruin theory
4. Feller (1968) — classical random walk results
5. Spitzer (1964) — random walk theory
6. Ferguson (1989) — secretary problem survey
7. Gilbert & Mosteller (1966) — secretary problem classic
8. Glasserman (2004) — Monte Carlo in finance
9. Karatzas & Shreve (1998) — mathematical finance
10. Black & Scholes (1973) — option pricing
11. Chow, Y.S. & Teicher, H. (1997). *Probability Theory*, 3rd ed. Springer. — for OST conditions and uniform integrability
12. Grimmett, G. & Stirzaker, D. (2001). *Probability and Random Processes*, 3rd ed. Oxford. — general reference

### Implementation
- Add entries to `paper/references.bib`
- Add `\cite{}` commands at appropriate locations in each chapter
- Update the text where citations are added to flow naturally

### Feasibility Verdict: **HIGH** — mechanical work of adding ~12 bib entries and inserting citations at appropriate locations. Estimated ~2 hours.
