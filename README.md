# Credit Default Risk Prediction

Модель предсказания дефолта клиента на данных Home Credit Default Risk.

## Результаты
- **ROC-AUC**: 0.7603
- **Brier Score** (после калибровки): 0.0667

## Что реализовано
- Разведочный анализ данных (EDA)
- Feature Engineering (отношения сумм, возраст, стаж и др.)
- Агрегация данных из `bureau` и `previous_application`
- Обучение LightGBM + подбор гиперпараметров (Optuna)
- Интерпретация модели через SHAP
- Калибровка вероятностей (Isotonic Regression)

- ## Основные инсайты
- Наиболее сильные признаки: `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`
- Молодые клиенты и клиенты с небольшим стажем имеют повышенный риск
- Созданные признаки-отношения показали высокую важность
- Калибровка значительно улучшила качество вероятностей

## Технологии
- Python, pandas, numpy
- LightGBM, scikit-learn
- Optuna, SHAP
- Matplotlib, Seaborn
