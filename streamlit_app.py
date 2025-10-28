import streamlit as st

st.title("Hello, World!")
st.write("This is a very simple Streamlit app.")

if st.button("Click me!"):
    st.write("Button was clicked!")
else:
    st.write("Click the button above.")