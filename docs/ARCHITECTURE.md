# System Architecture

**Developer:** Priyotosh Das | **Roll No:** 23053065  
**Project:** NOVA AI Data Analyst  

```mermaid
flowchart TD
    U[User] --> UI[Streamlit UI]
    UI --> R[Natural-language operation router]
    R --> P[Pandas analysis engine]
    P --> V[Verified result]
    V --> X[Local FLAN-T5 explanation]
    V --> C[Plotly visualization]
    V --> T[Supporting table]
    V --> UI
    Q[Data-quality profiler] --> UI
    I[Automated insight engine] --> UI
```
