import pandas as pd
import numpy as np
import joblib

def predict_deforestation_risk(Observation_Date,Region,District,Forest_Type,Forest_Area_ha,Tree_Cover_Loss_ha,Annual_Rainfall_mm,Population_Density,Fire_Incidents,Illegal_Logging):
    data = {
        "Observation_Date": Observation_Date,
        "Region": Region,
        "District": District,
        "Forest_Type": Forest_Type,
        "Forest_Area_ha": Forest_Area_ha,
        "Tree_Cover_Loss_ha": Tree_Cover_Loss_ha,
        "Annual_Rainfall_mm": Annual_Rainfall_mm,
        "Population_Density": Population_Density,
        "Fire_Incidents": Fire_Incidents,
        "Illegal_Logging": Illegal_Logging
    }

    df = pd.DataFrame([data])

    # Date
    df["Observation_Date"] = pd.to_datetime(df["Observation_Date"], errors="coerce")

    df["Year"] = df["Observation_Date"].dt.year
    df["Month"] = df["Observation_Date"].dt.month
    df["Day"] = df["Observation_Date"].dt.day

    df.drop(columns=["Observation_Date"], inplace=True)

    # Illegal_Logging
    df.rename(columns={"Illegal_Logging": "Illegal_Logging_Flag"}, inplace=True)
    df["Illegal_Logging_Flag"] = df["Illegal_Logging_Flag"].map({"Yes": 1, "No": 0})

    # One-Hot Encoding
    df = pd.get_dummies(
        df,
        columns=["Region", "District", "Forest_Type"],
        dtype=int
    )

    expected_columns = [
        "Forest_Area_ha",
        "Tree_Cover_Loss_ha",
        "Annual_Rainfall_mm",
        "Population_Density",
        "Fire_Incidents",
        "Illegal_Logging_Flag",
        "Year",
        "Month",
        "Day",
        "Region_Central",
        "Region_Eastern",
        "Region_Northern",
        "Region_Western",
        "District_Gulu",
        "District_Kabale",
        "District_Mbale",
        "District_Mbarara",
        "District_Mukono",
        "District_Wakiso",
        "Forest_Type_Dry Forest",
        "Forest_Type_Mangrove",
        "Forest_Type_Montane Forest",
        "Forest_Type_Plantation",
        "Forest_Type_Tropical Rainforest",
        "Forest_Type_Woodland",
    ]

    df = df.reindex(columns=expected_columns, fill_value=0)

    model = joblib.load("random_forest_model.pkl")

    prediction = model.predict(df)

    encoder = joblib.load('label_encoder.pkl')
    answer = encoder.inverse_transform(prediction)

    return answer[0]


# data = {
#     "Observation_Date": "2024-06-27",
#     "Region": "Central",
#     "District": "Wakiso",
#     "Forest_Type": "Mangrove",
#     "Forest_Area_ha": 4179.46,
#     "Tree_Cover_Loss_ha": 235.27,
#     "Annual_Rainfall_mm": 1358.1,
#     "Population_Density": 657,
#     "Fire_Incidents": 16,
#     "Illegal_Logging": "Yes"
# }

value = predict_deforestation_risk("2024-06-27", "Central", "Wakiso", "Mangrove", 4179.46, 235.27, 1358.1, 657, 16, "Yes")


print(f"Predicted Risk: {value} \nThanks!!!!!")