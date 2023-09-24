import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# 加载训练好的模型
with open("rsf.pickle", "rb") as f:
    rsf = pickle.load(f)

# 创建界面标题
st.title('Machine Learning to Evaluate Prognosis of DCM Patient')

# 创建输入表单
st.sidebar.header('Input Parameters')

# 创建下拉框ethnicity, Han Ethnicity代表1, Other Ethnicity代表0
ethnicity_mapping = {'Han Ethnicity': 1, 'Other Ethnicity': 0}
ethnicity = st.sidebar.selectbox('Ethnicity', list(ethnicity_mapping.keys()))
ethnicity_value = ethnicity_mapping[ethnicity]

age = st.sidebar.slider('Age (years)', 0, 100, 40)
qtc = st.sidebar.slider('QTC (ms)', 0, 1000, 450)
lvedd = st.sidebar.slider('LVEDD (mm)', 0, 150, 50)
lvef = st.sidebar.slider('LVEF (%)', 0, 100, 50)
alb = st.sidebar.slider('ALB (g/L)', 0, 100, 4)
ntpro_bnp = st.sidebar.slider('NTProBNP (ng/mL)', 0, 100000, 1000)

# 创建预测按钮
if st.sidebar.button('Predict'):

    st.write()

    # 创建输入数据的DataFrame
    input_data = pd.DataFrame({
        'ethnicity': [ethnicity_value],
        'age': [age],
        'QTC': [qtc],
        'LVEDD': [lvedd],
        'LVEF': [lvef],
        'ALB': [alb],
        'NTProBNP': [ntpro_bnp]
    })

    # 计算生存曲线
    surv = rsf.predict_survival_function(input_data, return_array=True)

    # 绘制生存曲线
    plt.figure()
    for i, s in enumerate(surv):
        plt.step(rsf.event_times_, s, where="post", label=str(i))

    
    plt.ylabel("Survival Probability")
    plt.xlabel("Time in months")
    plt.grid(True)


    rsf_score = rsf.predict(input_data)[0]
    # 预测值小于3.52时，显示低风险
    if rsf_score < 3.52:
        rsf_score = "Low Risk Group"
    # 预测值介于3.52-8.83时，显示中风险
    elif rsf_score >= 3.52 and rsf_score < 8.83:
        rsf_score = "Medium Risk Group"
    #预测值大于8.83时，显示高风险
    else:
        rsf_score = "High Risk Group"
    
    st.subheader("Hazard stratification: " +  rsf_score)
    st.pyplot(plt)  # 在Streamlit应用程序中显示绘制的图形
