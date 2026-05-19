import streamlit as st
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import (
    apriori,
    fpgrowth,
    association_rules
)

import matplotlib.pyplot as plt

# Page Config

st.set_page_config(
    page_title="Market Basket Analysis",
    layout="wide"
)

# Title

st.title("Market Basket Analysis App")

st.write(
    "Association Rule Mining using Apriori and FP-Growth Algorithms"
)

# Sidebar

st.sidebar.header("Settings")

min_support = st.sidebar.slider(
    "Minimum Support",
    0.1,
    1.0,
    0.2
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    0.1,
    1.0,
    0.6
)

# Sample Transaction Data

transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Diapers', 'Beer', 'Eggs'],
    ['Milk', 'Diapers', 'Beer', 'Cola'],
    ['Bread', 'Milk', 'Diapers', 'Beer'],
    ['Bread', 'Milk', 'Diapers', 'Cola'],
    ['Milk', 'Bread'],
    ['Bread', 'Butter'],
    ['Milk', 'Diapers', 'Beer'],
    ['Bread', 'Milk', 'Diapers'],
    ['Milk', 'Bread', 'Butter', 'Eggs'],
    ['Diapers', 'Beer'],
    ['Milk', 'Cola'],
    ['Bread', 'Cola'],
    ['Milk', 'Bread', 'Diapers', 'Beer'],
    ['Bread', 'Milk', 'Diapers', 'Butter'],
    ['Milk', 'Eggs'],
    ['Bread', 'Eggs'],
    ['Milk', 'Bread', 'Cola'],
    ['Diapers', 'Cola'],
    ['Bread', 'Diapers', 'Beer'],
    ['Milk', 'Bread', 'Diapers', 'Eggs'],
    ['Milk', 'Bread', 'Beer'],
    ['Bread', 'Butter', 'Cola'],
    ['Milk', 'Butter'],
    ['Bread', 'Milk', 'Diapers', 'Beer', 'Cola'],
    ['Milk', 'Diapers'],
    ['Bread', 'Diapers', 'Eggs'],
    ['Milk', 'Bread', 'Butter', 'Cola'],
    ['Beer', 'Cola'],
    ['Milk', 'Bread', 'Diapers', 'Beer'],
]

# Button

if st.button("Run Analysis"):

    # Transaction Data

    st.header("Transaction Data")

    transaction_df = pd.DataFrame({
        "Transactions": transactions
    })

    st.dataframe(transaction_df)

    # One Hot Encoding

    st.header("One-Hot Encoded Data")

    te = TransactionEncoder()

    te_array = te.fit(transactions).transform(transactions)

    df = pd.DataFrame(
        te_array,
        columns=te.columns_
    )

    st.dataframe(df.head())

    # Support

    st.header("Support Value")

    support_milk = df['Milk'].sum() / len(df)

    st.write(
        f"Support of Milk: {round(support_milk, 2)}"
    )

    # Thresholds

    st.header("Threshold Values")

    st.write(f"Minimum Support: {min_support}")
    st.write(f"Minimum Confidence: {min_confidence}")

    # Apriori

    st.header("Frequent Itemsets (Apriori)")

    frequent_itemsets = apriori(
        df,
        min_support=min_support,
        use_colnames=True
    )

    frequent_itemsets = frequent_itemsets.sort_values(
        by='support',
        ascending=False
    )

    st.dataframe(
        frequent_itemsets.head(10)
    )

    # FP Growth

    st.header("Frequent Itemsets (FP-Growth)")

    frequent_itemsets_fp = fpgrowth(
        df,
        min_support=min_support,
        use_colnames=True
    )

    frequent_itemsets_fp = frequent_itemsets_fp.sort_values(
        by='support',
        ascending=False
    )

    st.dataframe(
        frequent_itemsets_fp.head(10)
    )

    # Association Rules

    st.header("Association Rules")

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    if not rules.empty:

        st.dataframe(
            rules.head(10)
        )

        # Metrics

        st.header("Metrics")

        st.dataframe(
            rules[
                ['support', 'confidence', 'lift']
            ].head(10)
        )

        # Filtered Rules

        st.header("Filtered Rules")

        rules_filtered = rules[
            (rules['lift'] > 1) &
            (rules['confidence'] > 0.6)
        ]

        st.dataframe(
            rules_filtered.head(10)
        )

        # Sorted Rules

        st.header("Sorted Rules")

        rules_sorted = rules_filtered.sort_values(
            by='confidence',
            ascending=False
        )

        st.dataframe(
            rules_sorted.head(10)
        )

        # Rule Examples

        st.header("Rule Examples")

        for i, row in rules_sorted.head(3).iterrows():

            st.write(
                f"{set(row['antecedents'])} "
                f"→ "
                f"{set(row['consequents'])}"
            )

        # Multi-item Rules

        st.header("Multi-item Rules")

        multi_item_rules = rules_sorted[
            rules_sorted['antecedents'].apply(
                lambda x: len(x) > 1
            )
        ]

        st.dataframe(
            multi_item_rules.head(10)
        )

        # Visualization

        st.header("Visualization")

        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = ax.scatter(
            rules['support'],
            rules['confidence'],
            s=rules['lift'] * 150,
            alpha=0.7
        )

        ax.set_xlabel("Support")
        ax.set_ylabel("Confidence")

        ax.set_title(
            "Support vs Confidence"
        )

        ax.grid(True)

        for i, row in rules.head(5).iterrows():

            label = (
                f"{set(row['antecedents'])}"
                f" → "
                f"{set(row['consequents'])}"
            )

            ax.text(
                row['support'],
                row['confidence'],
                label,
                fontsize=8
            )

        st.pyplot(fig)

        # Lift Interpretation

        st.header("Lift Interpretation")

        st.dataframe(
            rules[
                ['antecedents', 'consequents', 'lift']
            ].head(10)
        )

        # Strong Rules

        st.header("Strong Rules")

        strong_rules = rules[
            rules['lift'] > 1
        ]

        st.dataframe(
            strong_rules.head(10)
        )

    else:

        st.warning(
            "No association rules generated."
        )

    # Item Support Ranking

    st.header("Item Support Ranking")

    item_support = df.sum() / len(df)

    st.dataframe(
        item_support.sort_values(
            ascending=False
        )
    )

    # Binary Matrix Shape

    st.header("Binary Matrix Shape")

    st.write(df.shape)

    # Final Message

    st.success(
        "Analysis Completed Successfully!"
    )
