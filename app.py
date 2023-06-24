import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Function to run the Apriori algorithm
def run_apriori(data, min_support, min_confidence):
    # Convert the data to a list of transactions with string items
    transactions = data.applymap(str).values.tolist()

    # Convert the transactions to transaction format
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Apply Apriori algorithm
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # Display frequent itemsets and association rules
    st.subheader("Frequent Itemsets")
    st.dataframe(frequent_itemsets)

    st.subheader("Association Rules")
    st.dataframe(rules)

# Streamlit app
def main():
    st.title(":red[Apriori Algorithm Implementation]")
    st.write(":blue[Upload a CSV file to run the Apriori algorithm on your dataset].")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file is not None:
        # Read the uploaded file as a DataFrame
        data = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.subheader("Uploaded Data")
        st.write(data)

        # Apriori parameters
        st.subheader("Apriori Parameters")
        min_support = st.slider("Minimum Support", 0.0, 1.0, 0.1, step=0.01)
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.5, step=0.01)

        # Run Apriori and display the results
        if st.button("Run Apriori"):
            run_apriori(data, min_support, min_confidence)

# Run the Streamlit app
if __name__ == "__main__":
    main()
