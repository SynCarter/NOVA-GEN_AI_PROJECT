import pandas as pd
from src.engine import analyze_question

def main():
    df=pd.read_csv("data/sample_sales.csv")
    tests=[
        ("Average unit price in East","aggregate.mean"),
        ("Total revenue in West","aggregate.sum"),
        ("Maximum revenue in North","aggregate.max"),
        ("Minimum unit price in South","aggregate.min"),
        ("Top 5 revenue records","ranking.nlargest"),
        ("What are the strongest correlations?","statistics.correlation"),
        ("How many missing values are there?","quality.missing"),
        ("How many duplicate rows are there?","quality.duplicates"),
    ]
    passed=0
    for q,expected in tests:
        r=analyze_question(df,q)
        ok=r["operation"].startswith(expected)
        print(("PASS" if ok else "FAIL"), q, "->", r["operation"])
        passed += ok
    print(f"\nTool/operation accuracy: {passed}/{len(tests)} = {passed/len(tests):.0%}")
if __name__=="__main__": main()
