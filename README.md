#### В 2023 я участововала в интересном конкурсе - [конкурсе красоты кода от Сбера](https://habr.com/ru/companies/sberbank/news/759496/) в Data Science секции 


### Описание проекта

Данный проект представляет собой модель машинного обучения, которая классифицирует пароли на надежные, средние и ненадежные. Для обучения модели использовался датасет с паролями, для которого были сгенерированы фичи.

### Структура проекта

- *config.py* - файл с переменными, необходимыми для работы проекта
- *preprocessing.py* - файл с функциями для подготовки датафрейма
- *train_model.py* - файл с функциями для обучения модели
- *predict.py* - файл с функциями для предсказания модели (и на тестовой выборке, и для введенного пароля)

### Процесс работы модели

1. Загрузка данных: сначала происходит выгрузка дата по ссылке. Если это невозможно, то берется локальная копия.
2. Подготовка данных: в рамках EDA выясняется, что в датасете нет дублей и пустых значений, а задача представляет собой мультиклассовую классификацию с дисбалансом классов. Далее генерируются бинарные фичи (от числовых пришлось отказаться, т.к. возникла проблема мультиколлинеарности).
3. Обучение модели: датасет разбивается на train и test выборки с использованием параметра stratify, чтобы сохранить пропорции классов при разбиении. Модель обучается на train выборке, с использованием stratified кросс-валидации (для сохранения пропорций классов) и выбранными метриками (_f1 score macro_ и _f1 score weighted_). Данные из тестовой выборки модель не видит
4. Для модели выбрана логистическая регрессия, т.к. она выдавала результат сопостоваимый с бустингами. Гиперпараметры были подобраны во время экпериментов + учтено то, что задача является мултиклассовой классификацией. 
5. Предсказание модели: подгружается .pkl файл модели и происходит предсказание значений на тестовой выборке и/или паролей из списка.

### Результаты
Выбранные метрики:

_f1 score macro_ - есть дисбаланс классов, и нам важен каждый класс

_f1 score weighted_ - т.к. задача является мультиклассовая классификация, интересно посмотреть, как работает с учетом веса каждого класса

Дополнительно просматривала precision и recall для каждого класса (из classification report) - чтобы посмотреть, как модель определяет каждый класс в отдельности

Модель показала хорошие результаты на тестовых данных:

- f1 weighted score = 0.9813
- f1 macro score = 0.9724

### Используемые библиотеки

- pandas
- sklearn
