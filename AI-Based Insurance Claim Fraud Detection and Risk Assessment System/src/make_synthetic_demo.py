"""TESTING ONLY: builds a fake file with the SAME columns as fraud_oracle.csv so the pipeline can be
smoke-tested without internet. Never report results from this file. Use the real Kaggle CSV."""
import numpy as np, pandas as pd
from config import RAW_CSV

rng = np.random.default_rng(0)
n = 15420
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
def ch(v, p=None): return rng.choice(v, n, p=p)
df = pd.DataFrame({
 "Month": ch(months), "WeekOfMonth": rng.integers(1,6,n), "DayOfWeek": ch(days),
 "Make": ch(["Pontiac","Toyota","Honda","Mazda","Chevrolet","Accura","Ford","VW"]),
 "AccidentArea": ch(["Urban","Rural"],[.9,.1]), "DayOfWeekClaimed": ch(days),
 "MonthClaimed": ch(months), "WeekOfMonthClaimed": rng.integers(1,6,n),
 "Sex": ch(["Male","Female"],[.85,.15]), "MaritalStatus": ch(["Married","Single","Divorced","Widow"],[.7,.28,.01,.01]),
 "Age": np.where(rng.random(n)<.02, 0, rng.integers(16,80,n)),
 "Fault": ch(["Policy Holder","Third Party"],[.7,.3]),
 "PolicyType": ch(["Sedan - Collision","Sedan - Liability","Sport - Collision","Utility - All Perils"]),
 "VehicleCategory": ch(["Sedan","Sport","Utility"],[.6,.35,.05]),
 "VehiclePrice": ch(["less than 20000","20000 to 29000","30000 to 39000","40000 to 59000","60000 to 69000","more than 69000"]),
 "PolicyNumber": np.arange(1,n+1), "RepNumber": rng.integers(1,17,n),
 "Deductible": ch([400,500,700,300],[.96,.01,.02,.01]), "DriverRating": rng.integers(1,5,n),
 "Days_Policy_Accident": ch(["more than 30","15 to 30","8 to 15","none"],[.97,.01,.01,.01]),
 "Days_Policy_Claim": ch(["more than 30","15 to 30","8 to 15","none"],[.99,.005,.003,.002]),
 "PastNumberOfClaims": ch(["none","1","2 to 4","more than 4"]),
 "AgeOfVehicle": ch(["3 years","6 years","7 years","more than 7","5 years","new","4 years","2 years"]),
 "AgeOfPolicyHolder": ch(["26 to 30","31 to 35","36 to 40","41 to 50","51 to 65","over 65","16 to 17","18 to 20","21 to 25"]),
 "PoliceReportFiled": ch(["No","Yes"],[.97,.03]), "WitnessPresent": ch(["No","Yes"],[.99,.01]),
 "AgentType": ch(["External","Internal"],[.98,.02]),
 "NumberOfSuppliments": ch(["none","1 to 2","3 to 5","more than 5"]),
 "AddressChange_Claim": ch(["no change","under 6 months","1 year","2 to 3 years","4 to 8 years"],[.9,.02,.02,.03,.03]),
 "NumberOfCars": ch(["1 vehicle","2 vehicles","3 to 4","5 to 8"],[.9,.07,.02,.01]),
 "Year": ch([1994,1995,1996]), "BasePolicy": ch(["Collision","Liability","All Perils"]),
})
df.loc[0, ["MonthClaimed","DayOfWeekClaimed"]] = "0"
risk = 0.03 + 0.05*(df.Fault=="Policy Holder") + 0.04*(df.BasePolicy=="All Perils") + 0.02*(df.PastNumberOfClaims=="more than 4")
df["FraudFound_P"] = (rng.random(n) < risk*0.9).astype(int)
cols = list(df.columns); cols.remove("FraudFound_P"); cols.insert(15, "FraudFound_P")
RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
df[cols].to_csv(RAW_CSV, index=False)
print("Synthetic demo written to", RAW_CSV, df.shape)
