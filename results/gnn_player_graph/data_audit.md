# Dynamic Oracle — GNN Player & Playing Style Data Audit

## 1. Executive Summary
- **Dataset Coverage**: Multi-year FIFA player database from FIFA 15 through FIFA 22 (2015–2022) with over 18,000 players per annual edition.
- **Explicit Style Labels**: The requested arcade/managerial text labels (`Creative Playmaker`, `Box-to-Box`, `Destroyer`, etc.) are **NOT explicitly stored** in EA Sports FIFA datasets.
- **Resolution**: We constructed 7 continuous, normalized **Derived Style Representations** directly from 29 underlying FIFA attribute and trait dimensions.

## 2. Node & Edge Definitions
- **Nodes**: 11 players per team, each represented by a 26-dimensional feature vector (10 core attributes, 8 positional encodings, 7 continuous style profiles, 1 quality scalar).
- **Edges**: Multi-relational $11 \times 11$ adjacency matrices incorporating Same Club ($E_1$), Shared Caps ($E_2$), Shared Minutes ($E_3$), Positional Proximity ($E_4$), Style Compatibility ($E_5$), and Role Complementarity ($E_6$).

## 3. Strict Temporal Integrity
- For any international match on date $T$, player ratings are drawn strictly from the FIFA edition $\text{year} \le T.\text{year}$.
- Shared caps and minutes are computed strictly from historical matches $t < T$, ensuring zero future leakage.