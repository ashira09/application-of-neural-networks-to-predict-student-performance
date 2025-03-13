# Литвинцев Степан Алексеевич
# Применение нейронных сетей для прогнозирования успеваемости студента
**Нейронная сеть** — математическая модель, а также её программное или аппаратное воплощение, построенная по принципу организации биологических нейронных сетей и обучаемая на основе определенных данных.
  
Нейронные сети применяются для многих целей: распознавание образов и классификация, принятие решений и управление, кластеризация и так далее. В рамках данного проекта предпологается применение нейронных сетей для прогнозирования успеваемости за первый семестр только поступивших в университет студентов.
  
Такой прогноз может позволить эффективно решать учебно-воспитатальные задачи. Например, в случае плохого прогноза предпринимать превентивные меры для его недопущения (каким-то образом стимулировать студентов к более усердному обучению или каким-то образом помочь студентам).
***
## Описание проекта
Данный проект состоит из двух основных этапов:
1. **Первый этап** - обучение нейросети на основе результатов вступительных экзаменов студентов.
2. **Второй этап** - создание программы, которая на вход принимает результаты вступительных экзаменов студентов, передает на вход обученной модели и выводит результат её предсказания.
***
## Входные данные
Входные данные представляют таблицу со следующими содержанием:
|Приоритет|Балл по математике|Балл по русскому языку|Балл по выбранному предмету|Балл за индивидуальные достижения|Наличие серебрянной медали|Наличие золотой медали|
|---------|------------------|----------------------|---------------------------|---------------------------------|--------------------------|----------------------|
|...      |...               |...                   |...                        |...                              |...                       |...                   |
1. **Приоритет** - приоритет выбранного направления для абитуриента, принимает целое значение значение от 1 до 5, соответственно, где 1 - высший приоритет, а 5 - низший.
2. **Балл по математике** - балл абитуриента за экзамен по матетматике, принимает целое значение от 0 до 100, где 0 - низший балл, а 100 - высший.
3. **Балл по русскому языка** - балл абитуриента за экзамен по русскому языку, принимает целое значение от 0 до 100, где 0 - низший балл, а 100 - высший.
4. **Балл по выбранному предмету** - балл абитуриента за экзамен по выбранному предмету, принимает целое значение от 0 до 100, где 0 - низший балл, а 100 - высший.
5. **Балл за индивидуальные достижения** - балл абитуриент за индивидуальные достижения, принимает целое значение от 0 до 47, где 0 - низший балл, а 47 - высший.
6. **Наличие серебрянной медали** - наличие у абитуриента серебрянной медали за окончание школы, принимает значение 0 или 1, где 0 - нет медали, а 1 - есть.
7. **Наличие серебрянной медали** - наличие у абитуриента золотой медали за окончание школы, принимает значение 0 или 1, где 0 - нет медали, а 1 - есть.
***
## Используемые технологии
- Язык программирования: Python.
- GUI-библиотека: PyQt.
- Библиотека для обучения нейросетей: TensorFlow, Scikit-learn, PyTorch, NumPy, Pandas.
