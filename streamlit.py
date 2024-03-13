import streamlit as st
import requests

def main():
    st.title("SMS/Mail Spam Classification")
    st.write("This is a spam classification application that uses machine learning to classify SMS messages as spam or ham.")
    user_input = st.text_input("Input Data")
    if st.button("Send"):
        data = {"message": user_input}

        response = requests.post("http://localhost:5000/predict", json=data)
        
        print(response)

        if response.status_code == 200:
            result = response.text
            st.success(f"Output:{result}")
        else:
            st.error("Error occured")

if __name__ == "__main__":
    main()