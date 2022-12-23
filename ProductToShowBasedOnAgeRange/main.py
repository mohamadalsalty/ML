import pandas as pd

if __name__ == "__main__":
    file = pd.read_csv('sales_data.csv')
    age_range = '26-35'
    products_sales = file[file['age_range'] == age_range]
    best_product = products_sales.loc[products_sales['sales'].idxmax()]['product']
    print(f'The best product to show for a client in the {age_range} age range is: {best_product}')