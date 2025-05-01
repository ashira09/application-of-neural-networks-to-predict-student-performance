import sys
from PySide6.QtWidgets import QPushButton, QApplication, QVBoxLayout, QDialog, QLabel, QCheckBox, QComboBox, QSpinBox, QHBoxLayout, QSizePolicy
from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from tensorflow import keras
import numpy as np

class Form(QDialog):

    def __init__(self, parent=None):
        super(Form, self).__init__(parent)
        
        layer = keras.layers.TFSMLayer('model', call_endpoint='serving_default')
        input_layer = keras.Input(shape = (11,))
        outputs = layer(input_layer)
        self.model = keras.Model(input_layer, outputs)

        self.label_gender = QLabel('Выберите пол:')
        self.combobox_gender = QComboBox()
        self.combobox_gender.addItem('Мужской')
        self.combobox_gender.addItem('Женский')
        self.label_priority = QLabel('Введите приоритет:')
        self.combobox_priority = QComboBox()
        self.combobox_priority.addItem('1')
        self.combobox_priority.addItem('2')
        self.combobox_priority.addItem('3')
        self.combobox_priority.addItem('4')
        self.combobox_priority.addItem('5')
        self.label_math_score = QLabel('Введите балл по математике:')
        self.spinbox_math_score = QSpinBox()
        self.spinbox_math_score.setMaximum(100)
        self.label_russian_language_score = QLabel('Введите балл по русскому языку:')
        self.spinbox_russian_language_score = QSpinBox()
        self.spinbox_russian_language_score.setMaximum(100)
        self.label_choosed_discipline_score = QLabel('Введите балл по выбранному предмету:')
        self.spinbox_choosed_discipline_score = QSpinBox()
        self.spinbox_choosed_discipline_score.setMaximum(100)
        self.label_individual_achivements_score = QLabel('Введите балл за индивидуальные достижения:')
        self.spinbox_individual_achivements_score = QSpinBox()
        self.spinbox_individual_achivements_score.setMaximum(47)
        self.checkbox_have_gold_medal = QCheckBox('Есть золотая медаль')
        self.checkbox_have_silver_medal = QCheckBox('Есть серебряная медаль')
        self.layout_button_and_result = QHBoxLayout()
        self.button = QPushButton("Предсказать")
        self.button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout_label_and_result = QVBoxLayout()
        self.label_prediction_result = QLabel('Результат предсказания:')
        self.value_prediction_result = QLabel('Ожидание предсказания...')
        self.layout_label_and_result.addWidget(self.label_prediction_result)
        self.layout_label_and_result.addWidget(self.value_prediction_result)
        self.layout_button_and_result.addWidget(self.button)
        self.layout_button_and_result.addLayout(self.layout_label_and_result)

        self.layout = QVBoxLayout()
        self.layout_gender = QVBoxLayout()
        self.layout_gender.addWidget(self.label_gender)
        self.layout_gender.addWidget(self.combobox_gender)
        self.layout_priority = QVBoxLayout()
        self.layout_priority.addWidget(self.label_priority)
        self.layout_priority.addWidget(self.combobox_priority)
        self.layout_math_score = QVBoxLayout()
        self.layout_math_score.addWidget(self.label_math_score)
        self.layout_math_score.addWidget(self.spinbox_math_score)
        self.layout_russian_language_score = QVBoxLayout()
        self.layout_russian_language_score.addWidget(self.label_russian_language_score)
        self.layout_russian_language_score.addWidget(self.spinbox_russian_language_score)
        self.layout_choosed_discipline_score = QVBoxLayout()
        self.layout_choosed_discipline_score.addWidget(self.label_choosed_discipline_score)
        self.layout_choosed_discipline_score.addWidget(self.spinbox_choosed_discipline_score)
        self.layout_individual_achivements_score = QVBoxLayout()
        self.layout_individual_achivements_score.addWidget(self.label_individual_achivements_score)
        self.layout_individual_achivements_score.addWidget(self.spinbox_individual_achivements_score)
        self.layout_have_gold_medal = QVBoxLayout()
        self.layout_have_gold_medal.addWidget(self.checkbox_have_gold_medal)
        self.layout_have_silver_medal = QVBoxLayout()
        self.layout_have_silver_medal.addWidget(self.checkbox_have_silver_medal)
        self.layout.addLayout(self.layout_gender)
        self.layout.addLayout(self.layout_priority)
        self.layout.addLayout(self.layout_math_score)
        self.layout.addLayout(self.layout_russian_language_score)
        self.layout.addLayout(self.layout_choosed_discipline_score)
        self.layout.addLayout(self.layout_individual_achivements_score)
        self.layout.addLayout(self.layout_have_gold_medal)
        self.layout.addLayout(self.layout_have_silver_medal)
        self.layout.addLayout(self.layout_button_and_result)

        self.setLayout(self.layout)

        self.button.clicked.connect(self.predict)

        self.timer = QTimer()
        self.timer.setInterval(5000)
        self.timer.timeout.connect(self.clearPredictionResult)

    def predict(self):
        gender = 0 if (self.combobox_gender.currentText() == 'Мужской') else 1
        priority = int(self.combobox_priority.currentText())
        math_score = int(self.spinbox_math_score.value())
        russian_language_score = int(self.spinbox_russian_language_score.value())
        choosed_discipline_score = int(self.spinbox_choosed_discipline_score.value())
        achievements_score = int(self.spinbox_individual_achivements_score.value())
        have_gold_medal = int(self.checkbox_have_gold_medal.isChecked())
        have_silver_medal = int(self.checkbox_have_silver_medal.isChecked())
        average_discipline_score = float(math_score+russian_language_score+choosed_discipline_score)/3.
        sum_all_scores = math_score+russian_language_score+choosed_discipline_score+achievements_score
        percentage = (float(sum_all_scores)/347.)*100
        
        prediction_number = np.argmax(self.model.predict(np.array([[gender, priority, math_score, russian_language_score, choosed_discipline_score, achievements_score, have_gold_medal, have_silver_medal, average_discipline_score, sum_all_scores, percentage]]))['dense_2'])
        
        prediction_number_to_word = {0: 'Неудовлетворительно', 1: 'Удовлетворительно', 2: 'Хорошо', 3:'Отлично'}
        prediction_number_to_color = {0: 'red', 1: 'orange', 2: 'yellow', 3: 'green'}
        
        self.value_prediction_result.setText(f'<font color={prediction_number_to_color[prediction_number]}>{prediction_number_to_word[prediction_number]}</font>')
        self.timer.start()

    def clearPredictionResult(self):
        self.value_prediction_result.setText('Ожидание предсказания...')
        self.timer.stop()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app_icon = QIcon('app_icon.png')
    app.setApplicationDisplayName('Предсказание успеваемость студентов')
    app.setWindowIcon(app_icon)
    form = Form()
    form.show()
    sys.exit(app.exec())
