import streamlit as st
import mysql.connector
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt

# Fetch orders data from MySQL
@st.cache_data
def fetch_orders_data():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="pos_7"
    )

    query = """
        SELECT ID, order_date, ship_date, customer_id, product_id, product_name FROM orders
    """

    cursor = conn.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    conn.close()

    columns = ['ID', 'order_date', 'ship_date', 'customer_id', 'product_id', 'product_name']
    orders_df = pd.DataFrame(data, columns=columns)
    
    return orders_df

def market_basket_analysis(orders_df):
    # Clone the DataFrame to avoid modifying the cached object
    orders_df = orders_df.copy()
    
    # Convert order_date to datetime
    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])

    # Aggregate products by customer to form transactions
    transactions = orders_df.groupby(['customer_id'])['product_name'].apply(list).reset_index()

    # Prepare data for MBA
    transaction_list = transactions['product_name'].tolist()

    # Transform the transaction data
    transaction_encoder = TransactionEncoder()
    transaction_encoder_ary = transaction_encoder.fit(transaction_list).transform(transaction_list)
    transaction_df = pd.DataFrame(transaction_encoder_ary, columns=transaction_encoder.columns_)

    # Apply Apriori algorithm to find frequent itemsets
    frequent_itemsets = apriori(transaction_df, min_support=0.05, use_colnames=True)

    # Generate association rules
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)

        # Filter out rules with confidence of 1 to avoid infinite conviction
        rules = rules[rules['confidence'] < 1]

        # Add count of occurrences
        rules['count'] = rules.apply(lambda row: sum(transaction_df[list(row['antecedents'])].all(axis=1) & transaction_df[list(row['consequents'])].all(axis=1)), axis=1)

        # Create a column for the basket pair
        rules['basket_pair'] = rules['antecedents'].apply(lambda x: ', '.join([f"{item}" for item in list(x)])) + " -> " + rules['consequents'].apply(lambda x: ', '.join(list(x)))

        print("Association Rules Head:")
        print(rules.head())

        return rules
    else:
        return pd.DataFrame()

# Streamlit app
st.title("Market Basket Analysis")

orders_df = fetch_orders_data()
st.write("Orders Data", orders_df)

if st.button("Run Market Basket Analysis"):
    rules = market_basket_analysis(orders_df)
    
    if not rules.empty:
        # Select relevant columns for display
        rules_display = rules[['basket_pair', 'support', 'confidence', 'lift', 'count']]
        st.write("Association Rules", rules_display)
        
        # Create horizontal bar chart for the top 10 rules by lift
        top_rules = rules.nlargest(10, 'lift')
        fig, ax = plt.subplots()
        colors = plt.cm.tab20.colors
        top_rules.plot(kind='barh', x='basket_pair', y='lift', ax=ax, color=colors, legend=False)
        ax.set_title('Top 10 Association Rules by Lift')
        ax.set_xlabel('Lift')
        ax.set_ylabel('Rule')
        ax.set_xlim(0, 2.0)  # Set x-axis limit to 2.0
        ax.set_yticklabels(top_rules['basket_pair'], fontsize=10)  # Adjust y-axis labels
        st.pyplot(fig)
        st.write("Horizontal bar chart generated successfully.")
        
        # Display MBA insights
        st.subheader("MBA Insights")
        insights = []
        count = 0
        for _, rule in rules.nlargest(20, 'lift').iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            for ant in antecedents:
                for cons in consequents:
                    insight = f"Customers who usually buy **{ant}** are more likely to buy **{cons}**"
                    if insight not in insights:
                        insights.append(insight)
                        count += 1
                        st.write(f"{count}. {insight}", unsafe_allow_html=True)
                        if count == 5:
                            break
                if count == 5:
                    break
            if count == 5:
                break
    else:
        st.write("No association rules found. Try lowering the min_threshold value.")


