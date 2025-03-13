# Литвинцев Степан Алексеевич
# Применение нейронных сетей для прогнозирования успеваемости студента
**Нейронная сеть** — математическая модель, а также её программное или аппаратное воплощение, построенная по принципу организации биологических нейронных сетей и обучаемая на основе определенных данных.
  
Нейронные сети применяются для многих целей: распознавание образов и классификация, принятие решений и управление, кластеризация и так далее. В рамках данного проекта предпологается применение нейронных сетей для прогнозирования успеваемости за первый семестр только поступивших в университет студентов.
  
Такой прогноз может позволить эффективно решать учебно-воспитатальные задачи. Например, в случае плохого прогноза предпринимать превентивные меры для его недопущения (каким-то образом стимулировать студентов к более усердному обучению или каким-то образом помочь студентам).
***
## Описание проекта
Данный проект состоит из двух основных этапов:
1. Первый этап - обучение нейросети на основе результатов вступительных экзаменов студентов.
2. Второй этап - создание программы, которая на вход принимает результаты вступительных экзаменов студентов, передает на вход обученной модели и выводит результат её предсказания.
***
## Данные
|Приоритет|Балл по математике|Балл по русскому языку|Балл по выбранному предмету|Индивидуальные достижения|Наличие серебрянной медали|Наличие золотой медали|
|---------|------------------|----------------------|---------------------------|-------------------------|--------------------------|----------------------|
|...      |...               |...                   |...                        |...                      |...                       |...                   |
1. Приоритет - абитуриент мог подать документы на от 1 на 5 направлений обучения, каждому из которых присвоив приоритет от 1 до 5, соответственно, где 1 - высший приоритет, а 5 - низший.
## Используемые технологии
- Язык программирования: Python.
- GUI-библиотека: PyQt.
- Библиотека для обучения нейросетей: TensorFlow.
