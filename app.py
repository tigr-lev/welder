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

            # Загружаем модель для предсказания ширины шва (линейная регрессия)
            with open('lr_width_model.pkl', 'rb') as f:
                lr_model = pickle.load(f)

            # Загружаем модель для предсказания глубины шва (случайный лес)
            with open('rf_depth_model.pkl', 'rb') as f:
                rf_model = pickle.load(f)

            # Формируем данные для предсказания
            input_data = [[IW, IF, VW, FP]]

            # Предсказание ширины шва с помощью линейной регрессии
            predicted_width = lr_model.predict(input_data)

            # Предсказание глубины шва с помощью случайного леса
            predicted_depth = rf_model.predict(input_data)

            # Возвращаем результаты предсказаний в шаблон
            return render_template('main.html', 
                                   predicted_width=predicted_width[0], 
                                   predicted_depth=predicted_depth[0])
        
        except KeyError as e:
            return render_template('main.html', error=f"Отсутствует поле: {str(e)}")
        except ValueError as e:
            return render_template('main.html', error=f"Недопустимое значение: {str(e)}")
        except Exception as e:
            return render_template('main.html', error=f"Ошибка: {str(e)}")

if __name__ == '__main__':  # Запуск приложения
    app.run(debug=True)  # Включаем режим отладки для лучшего вывода ошибок