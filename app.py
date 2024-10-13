# -*- coding: utf-8 -*-
import flask
from flask import Flask, render_template, request
import pickle
from sklearn.linear_model import LinearRegression

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['POST', 'GET'])
@app.route('/index', methods=['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')
    
    if flask.request.method == 'POST':
        try:
            # Используем большие буквы для переменных
            IW = float(flask.request.form['IW'])  # Получаем значение IW
            IF = float(flask.request.form['IF'])   # Получаем значение IF
            VW = float(flask.request.form['VW'])   # Получаем значение VW
            FP = float(flask.request.form['FP'])   # Получаем значение FP

            with open('lr_model.pkl', 'rb') as f:
                loaded_model = pickle.load(f)

            # Используем все четыре переменные для предсказания
            y_pred = loaded_model.predict([[IW, IF, VW, FP]])  # Предсказание на основе всех четырех параметров

            return render_template('main.html', result=y_pred[0])  # Передаем результат в шаблон
        except KeyError as e:
            return render_template('main.html', error=f"Отсутствует поле: {str(e)}")
        except ValueError as e:
            return render_template('main.html', error=f"Недопустимое значение: {str(e)}")
        except Exception as e:
            return render_template('main.html', error=f"Ошибка: {str(e)}")

if __name__ == '__main__':  # Запуск приложения
    app.run(debug=True)  # Включаем режим отладки для лучшего вывода ошибок