from flask import Flask, request, render_template
from flask import send_file
import pickle
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


app = Flask(__name__)

# Modelni yuklash
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Home sahifa


@app.route('/download')
def download():
    file_path = 'data.csv'
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "CSV fayl topilmadi."


@app.route('/', methods=['GET', 'POST'])
def index():
    price = None
    if request.method == 'POST':
        data = [
            int(request.form['OverallQual']),
            int(request.form['GrLivArea']),
            int(request.form['GarageArea']),
            int(request.form['TotalBsmtSF']),
            int(request.form['FullBath']),
            int(request.form['YearBuilt']),
        ]

        # DataFrame shaklida kiritamiz (ogohlantirish chiqmasligi uchun)
        df_input = pd.DataFrame([data], columns=['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt'])
        price = model.predict(df_input)[0]
        price = round(price, 2)

        # CSVga yozish
        with open('data.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            if os.stat('data.csv').st_size == 0:
                writer.writerow(['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'PredictedPrice'])
            writer.writerow(data + [price])

    return render_template('index.html', price=price)

# Grafik sahifa
@app.route('/chart')
def chart():
    try:
        if not os.path.exists('data.csv'):
            return "⚠️ Grafik mavjud emas. Iltimos, formani to‘ldiring."

        df = pd.read_csv('data.csv')

        if df.empty:
            return "⚠️ CSV fayl bo‘sh. Grafik chizish uchun ma'lumot yo‘q."

        plt.figure(figsize=(10, 6))
        plt.style.use('ggplot')  # Yoki seaborn o‘rniga
        plt.plot(
            df['YearBuilt'], df['PredictedPrice'],
            marker='o', linestyle='-', linewidth=2,
            color='#1f77b4', label='Narx ($)'
        )

        plt.title('🧱 Qurilgan yilga qarab uy narxlari', fontsize=16, fontweight='bold')
        plt.xlabel('Qurilgan yili', fontsize=13)
        plt.ylabel('Narx ($)', fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()

        # Faylni saqlash
        if not os.path.exists('static'):
            os.makedirs('static')

        plt.savefig('static/chart.png')
        plt.close()

        return render_template('chart.html')
    
    except Exception as e:
        return f"❌ Xatolik yuz berdi: {str(e)}"
@app.route('/results')
def results():
    if not os.path.exists('data.csv'):
        return "⛔ Hali hech qanday bashorat mavjud emas."

    df = pd.read_csv('data.csv')

    # Ma'lumotlar mos formatga keltiriladi
    data = []
    for _, row in df.iterrows():
        data.append({
            'rooms': row['OverallQual'],
            'area': row['GrLivArea'],
            'age': 2025 - row['YearBuilt'],  # Yoshlanish
            'predicted_price': round(row['PredictedPrice'], 2)
        })

    return render_template('results.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)