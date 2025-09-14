import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load saved instances (scaler, pca, model)
# -----------------------------
with open('scaler.joblib', 'rb') as file:
    scale = joblib.load(file)

with open('pca.joblib', 'rb') as file:
    pca = joblib.load(file)

with open('final_model.joblib', 'rb') as file:
    model = joblib.load(file)


# -----------------------------
# Prediction function
# -----------------------------
def prediction(input_list):
    scaled_input = scale.transform([input_list])
    pca_input = pca.transform(scaled_input)
    output = model.predict(pca_input)[0]

    if output == 0:
        return 'Developed'
    elif output == 1:
        return 'Underdeveloped'
    else:
        return 'Developing'


# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.title("🌟 𝓗𝓔𝓛𝓟 𝓝𝓖𝓞 𝓕𝓞𝓤𝓝𝓓𝓐𝓣𝓘𝓞𝓝 🌟")
    st.subheader("🌈 Humanity | Empowerment | Hope 🌈 ❤️ Strong Impact")

    # Show NGO image
    st.image(
        r"C:\Users\saurav\GREATLEARNING FEB 25\usl machine learning 3\Case Study\Deployment\Gemini_Generated_Image_4j734y4j734y4j73.png",
        caption="Help NGO Foundation",
        use_column_width=True
    )

    # Inputs
    gdp = st.text_input('Enter the GDP per Population of a country')
    inc = st.text_input('Enter the per capita income of a country')
    imp = st.text_input('Enter the Imports in terms of % of GDP')
    exp = st.text_input('Enter the Exports in terms of % of GDP')
    inf = st.text_input('Enter the inflation rate in a country (%)')

    hel = st.text_input('Enter the expenditure on health in terms % of GDP')
    ch_m = st.text_input('Enter the no of deaths per 1000 births for <5 yrs')
    fer = st.text_input('Enter the avg children born to a woman in a country')
    lf = st.text_input('Enter the average life expectancy in a country')

    # Predict button
    if st.button('Predict'):
        try:
            in_data = [
                float(ch_m), float(exp), float(hel), float(imp),
                float(inc), float(inf), float(lf), float(fer), float(gdp)
            ]
            response = prediction(in_data)
            st.success(f"The country is classified as: {response}")
        except ValueError:
            st.error("⚠️ Please enter valid numeric values for all fields.")


# -----------------------------
# Run the app
# -----------------------------
if __name__ == '__main__':
    main()
