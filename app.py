import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# 加载训练好的模型
with open("rsf.pickle", "rb") as f:
    EST = pickle.load(f)

# 创建界面标题
st.title('Machine Learning to Evaluate Prognosis of DCM Patient')

# 创建输入表单
st.sidebar.header('Input Parameters')

# 创建下拉框ethnicity, Han Ethnicity代表1, Other Ethnicity代表0
ethnicity_mapping = {'Han Ethnicity': 1, 'Other Ethnicity': 0}
ethnicity = st.sidebar.selectbox('Ethnicity', list(ethnicity_mapping.keys()))
ethnicity_value = ethnicity_mapping[ethnicity]

drinking_mapping = {'Yes': 1, 'No': 0}
drinking = st.sidebar.selectbox('drinking', list(drinking_mapping.keys()))
drinking_value = drinking_mapping[drinking]

AF_mapping = {'Yes': 1, 'No': 0}
AF = st.sidebar.selectbox('AF', list(AF_mapping.keys()))
AF_value = AF_mapping[AF]

Digoxin_mapping = {'Yes': 1, 'No': 0}
Digoxin = st.sidebar.selectbox('Digoxin', list(Digoxin_mapping.keys()))
Digoxin_value = AF_mapping[AF]

age = st.sidebar.slider('Age (years)', 0, 100, 40)
SBP = st.sidebar.slider('SBP (years)', 50, 200, 40)
BMI = st.sidebar.slider('BMI (Kg/m\u00B2)', 10, 60, 40)
lvedd = st.sidebar.slider('LVEDD (mm)', 0, 150, 50)
lvef = st.sidebar.slider('LVEF (%)', 45, 100, 50)
QRSL = st.sidebar.slider('QRS (ms)', 200, 500, 450)
Lymph = st.sidebar.slider('Lym (10⁹/L)', 0, 30, 450)
Na = st.sidebar.slider('Na⁺ (g/L)', 50, 200, 4)
Glu = st.sidebar.slider('FBG (mmol/L)', 0, 50, 4)
ntpro_bnp = st.sidebar.slider('NT-proBNP (ng/mL)', 0, 100000, 1000)


# 创建预测按钮
if st.sidebar.button('Predict'):

    st.write()

    # 创建输入数据的DataFrame
    input_data = pd.DataFrame({
        'ethnicity': [ethnicity_value],
        'drinking': [drinking_value],
        'AF': [AF_value],
        'Digoxin': [Digoxin_value],
        'age': [age],
        'SBP': [SBP],
        'BMI': [BMI],
        'LVEDD': [lvedd],
        'LVEF': [lvef],
        'QRSL': [QRSL],
        'Lymph': [Lymph],
        'Na': [Na],
        'Glu': [Glu],
        'NTProBNP': [ntpro_bnp]
    })

    # 计算生存曲线
    surv = EST.predict_survival_function(input_data, return_array=True)

    # 绘制生存曲线
    plt.figure()
    for i, s in enumerate(surv):
        plt.step(EST.event_times_, s, where="post", label=str(i))

    
    plt.ylabel("Survival Probability")
    plt.xlabel("Time in months")
    plt.grid(True)


    EST_score = EST.predict(input_data)[0]
    # 预测值小于5.01时，显示低风险
    if EST_score < 5.01:
        EST_score = "Low Risk Group"
    # 预测值介于5.01-9.65时，显示中风险
    elif EST_score >= 5.01 and EST_score < 9.65:
        EST_score = "Medium Risk Group"
    #预测值大于9.65时，显示高风险
    else:
        EST_score = "High Risk Group"
    
    st.subheader("Hazard stratification: " +  EST_score)
    st.pyplot(plt)  # 在Streamlit应用程序中显示绘制的图形