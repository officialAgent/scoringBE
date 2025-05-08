import pandas as pd
import numpy as np
import random

def generate_credit_scoring_csv(rows: int, filename: str = "data/credit_scoring_data.csv"):
        device_prices = {
            'Samsung S21': 700,
            'Samsung S22': 800,
            'iPhone 13': 900,
            'iPhone 14': 1100,
            'Xiaomi Mi 11': 600,
            'Huawei P50': 650
        }

        device_types = list(device_prices.keys())
        data = []

        customer_data_map = {}  # Store existing customer data

        for _ in range(rows):
            customer_id = random.randint(10000, 99999)  # bigger pool for IDs

            if customer_id in customer_data_map:
                # Reuse existing customer data
                age, gender, customer_duration = customer_data_map[customer_id]
            else:
                age = int(np.clip(np.random.normal(35, 10), 18, 70))
                gender = random.choice(['F', 'M'])
                max_duration = (age - 17) * 12
                customer_duration = random.randint(1, max(6, max_duration))
                customer_data_map[customer_id] = (age, gender, customer_duration)

            # Device selection
            device = random.choices(
                population=device_types,
                weights=[15, 20, 25, 15, 15, 10],
                k=1
            )[0]
            device_price = device_prices[device]

            # Monthly bill
            base_bill = np.random.normal(45, 20)
            avg_monthly_bill = int(np.clip(base_bill, 10, 150))

            # Billing delays
            if customer_duration < 12:
                billing_delay_count = np.random.poisson(2)
            elif customer_duration > 36:
                billing_delay_count = np.random.poisson(0.5)
            else:
                billing_delay_count = np.random.poisson(1)
            billing_delay_count = int(np.clip(billing_delay_count, 0, 5))

            # Payment delays
            if random.random() < 0.05:  # 5% bad customers
                payment_delay_count = random.randint(5, 10)
                payment_delay_total_days = payment_delay_count * random.randint(10, 30)
            else:
                payment_delay_count = random.randint(0, 6)
                payment_delay_total_days = payment_delay_count * random.randint(1, 15)

            # CrifScore logic — now based on behavior
            if customer_duration > 30 and random.random() > 0.3:
                crif_score = np.nan
            else:
                if billing_delay_count == 0 and payment_delay_count == 0:
                    crif_score = random.randint(70, 100)
                elif billing_delay_count <= 2 and payment_delay_count <= 2:
                    crif_score = random.randint(40, 80)
                else:
                    crif_score = random.randint(0, 50)

            # Loan logic
            if random.random() < 0.1:  # 10% chance
                max_loan = int(device_price)  # Full price as max loan
            else:
                if not np.isnan(crif_score):
                    risk_factor = (100 - crif_score) / 100
                    max_loan = int(device_price * (0.5 + 0.4 * (1 - risk_factor)))
                else:
                    max_loan = int(device_price * 0.7)

            loan_price = random.randint(int(device_price * 0.5), min(max_loan, int(device_price * 1.1)))

            loan_duration = random.choice([12, 18, 24])
            monthly_installment = round(loan_price / loan_duration, 2)

            loan_status = 'closed'

            # Risk calculation — new softer bonus based on base crif risk
            if not np.isnan(crif_score):
                base_risk = (100 - crif_score) / 100
                bonus_risk = (billing_delay_count / 5) * (1 - base_risk) * 0.4
                bonus_risk += (payment_delay_total_days / 90) * (1 - base_risk) * 0.2
                risk = base_risk + bonus_risk
            else:
                risk = (billing_delay_count / 5) * 0.6 + (payment_delay_total_days / 90) * 0.4

            risk = np.clip(risk, 0, 1)

            loan_settled = random.choices([1, 0], weights=[1 - risk, risk])[0]

            data.append([
                customer_id, age, gender, customer_duration,
                avg_monthly_bill, billing_delay_count,
                crif_score, device, device_price, loan_price,
                loan_duration, monthly_installment,
                payment_delay_count, payment_delay_total_days,
                loan_status, loan_settled
            ])

        columns = [
            'Customer_ID', 'Age', 'Gender', 'Customer_Duration',
            'Avg_Monthly_Bill_EUR', 'Billing_Delay_Count',
            'crifScore', 'Device_Type', 'Device_Price_EUR', 'Loan_Price_EUR',
            'Loan_Duration', 'Monthly_Installment_EUR', 'Payment_Delay_Count',
            'Payment_Delay_Total_Days', 'Loan_Status', 'Loan_Settled'
        ]

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"{filename} generated with {rows} rows.")

        # Print Loan_Settled statistics
        settled_counts = df['Loan_Settled'].value_counts().sort_index()
        total = settled_counts.sum()

        print("\nLoan_Settled Statistics:")
        for val, count in settled_counts.items():
            percentage = (count / total) * 100
            status = "Settled" if val == 1 else "Defaulted"
            print(f"  {status} ({val}): {count} loans ({percentage:.2f}%)")



# Example call



