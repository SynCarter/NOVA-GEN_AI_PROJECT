import pandas as pd
from src.engine import analyze_question

def test_filtered_average():
    df=pd.DataFrame({"region":["East","East","West"],"unit_price":[10,30,100]})
    r=analyze_question(df,"average unit price in East")
    assert "20.00" in r["answer"]

def test_filtered_total():
    df=pd.DataFrame({"region":["East","West"],"revenue":[100,200]})
    r=analyze_question(df,"total revenue in East")
    assert "100.00" in r["answer"]

def test_missing():
    df=pd.DataFrame({"x":[1,None,3]})
    r=analyze_question(df,"how many missing values")
    assert "1" in r["answer"]

def test_duplicates():
    df=pd.DataFrame({"x":[1,1,2]})
    r=analyze_question(df,"duplicate rows")
    assert "1" in r["answer"]

def test_correlation():
    df=pd.DataFrame({"a":[1,2,3],"b":[2,4,6]})
    r=analyze_question(df,"strongest correlations")
    assert "1.000" in r["answer"]
