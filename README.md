# UR3 actuator sizing + динамические тесты

Репозиторий для подбора приводов UR3 и проверки нагрузок на суставы через симулятор MuJoCo.

## Структура

```text
models/
  source/
    description/
      universalUR3.urdf      # исходный URDF UR3
      universalUR3.xml       # базовая MJCF-модель UR3
    meshes/ur3/...           # геометрия, только для визуализации
    my/                      # сюда генерируются MJCF с новыми моторами
      ur3_new_replace_motor_single.xml
      ur3_new_variant_01.xml
      ...
  motors/
    old_motors.csv           # параметры предполагаемых моторов UR3
    new_motors.csv           # варианты редукторов и новых приводов

src/
  ur3_desc_genetator.py      # генератор новых MJCF по CSV
  ur3_tests.py               # статические и динамические тесты
````

## Зависимости

Рекомендуется отдельное виртуальное окружение:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install mujoco matplotlib seaborn lxml
```

## 1. Генерация моделей с новыми моторами

Скрипт `ur3_desc_genetator.py`:

* читает базовую модель `models/source/description/universalUR3.xml`,
* читает параметры моторов из `models/motors/old_motors.csv` и `models/motors/new_motors.csv`,
* генерирует варианты MJCF в `models/source/my/`.

Запуск (из корня проекта):

```bash
python src/ur3_desc_genetator.py
```

После этого в `models/source/my/` появляются файлы `ur3_new_variant_XX.xml` и эталон `ur3_new_replace_motor_single.xml`.

## 2. Тесты в MuJoCo

Скрипт `ur3_tests.py` умеет запускать три теста:

* `static`    — статические моменты с 3 кг грузом на `ee_link`;
* `cartesian` — динамика по  траектории в декартовом пространстве;
* `joint`     — динамика по  траектории в joint-space.

Общий формат запуска:

```bash
python src/ur3_tests.py --test {static|cartesian|joint} --xml ПУТЬ_К_XML
```

Примеры:

```bash
# 1) Статика для базовой модели UR3
python src/ur3_tests.py --test static --xml models/source/description/universalUR3.xml

# 2) Динамика в декартовых для одной из новых конфигураций моторов
python src/ur3_tests.py --test cartesian --xml models/source/my/ur3_new_variant_01.xml

# 3) Динамика в joint-space для эталонной конфигурации
python src/ur3_tests.py --test joint --xml models/source/my/ur3_new_replace_motor_single.xml
```

Что рисуется в динамических тестах:

* графики моментов по времени для всех суставов;
* 3D-траектория конца звена;
* violin-плотности моментов по суставам;
* диаграмма Torque–Speed;
* Performance Envelope с трейсовыми кривыми (|ω_j(t)|, |τ_j(t)|) для каждого сустава.


По результатам графиков определяется, подходят те или иные моторы для данного манипулятора
