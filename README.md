# Deep-Live-Cam — live fork

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/onnxruntime--gpu-1.26-green.svg)](https://onnxruntime.ai/)
[![PySide6](https://img.shields.io/badge/UI-PySide6-41cd52.svg)](https://doc.qt.io/qtforpython/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-lightgrey.svg)](LICENSE)

🇬🇧 [English](#english) · 🇷🇺 [Русский](#русский)

---

## English

A fork of [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) focused on the **live webcam** path: photo-based face swap that holds its shape under fast motion, fingers and turns, with per-person calibration and a display pipeline that does not lag behind the head.

### Features

- ⚡ **CUDA-graph inference** for the detector, landmarks, swappers and enhancers; XSeg occluder rebuilt for the GPU (8× faster)
- 🎯 **Optical-flow face tracking** between detections: forty support points over the head, forward-backward check, RANSAC similarity — a finger or a mic does not bend the alignment
- 🧭 **Per-person calibration** (one minute in front of the camera): reference outline, expression-proof crop, fade to the real face past your own turn limits
- 👁 **Real blinks and eyes** — the swap models only squint; the eye region is handed back to the camera during a blink, optionally always
- 👄 **Mouth mask** bounded by facial features (lips → nose base → chin), so a moustache is never cut in half
- 🕒 **Low-latency reprojection** — the last finished swap is moved onto the newest camera frame, so a heavy enhancer delays the expression, not the position
- 🧩 Swappers: Inswapper-128, HyperSwap 1a/1b/1c (256); enhancers: GPEN-256/512/1024, GFPGAN 1.4, CodeFormer, RestoreFormer++
- 🖥 Responsive PySide6 UI (Models / Mask / Motion / Output tabs), English and Russian

### Requirements

| | |
|---|---|
| OS | Windows 10/11 (CUDA), macOS Apple Silicon (CoreML) |
| Python | 3.12 |
| GPU | NVIDIA with CUDA 12 for the live path (tested on RTX 5060 Ti 16 GB) |
| Models | downloaded on first use into `models/` |

### Install

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
python run.py --execution-provider cuda
```

`ffmpeg` must be on `PATH` (or next to `run.py`) for video output.

### Docs

- [docs/calibration.md](docs/calibration.md) — profiles, pose proxies, expression-proof crop, eye and mouth reveal
- [docs/reprojection.md](docs/reprojection.md) — the display-side tracker and why it never trusts a half-applied correction
- [docs/tracking.md](docs/tracking.md) — the face tracker, support points, hold through missed detections

<details>
<summary>Project layout</summary>

```
modules/
  ui.py                 PySide6 main window and live loop
  ui_calibration.py     guided calibration dialog
  calibration.py        profiles, pose proxies, stable keypoints
  face_tracker.py       optical-flow tracker
  reprojection.py       late reprojection onto the newest frame
  face_reveal.py        eye / mouth reveal masks
  face_occluder.py      XSeg occlusion mask
  cuda_graph.py         CUDA graph sessions
  processors/frame/     swappers and enhancers
locales/                UI translations
docs/                   design notes
```
</details>

### License

AGPL-3.0, as upstream. Models keep their own licenses; the swap models are for non-commercial research use.

---

## Русский

Форк [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam), заточенный под **живую камеру**: подмена лица по фотографии, которая держит форму при резких движениях, пальцах и поворотах, с калибровкой под человека и конвейером отображения, который не отстаёт от головы.

### Возможности

- ⚡ **CUDA-графы** для детектора, точек, свапперов и улучшателей; окклюдер XSeg пересобран под GPU (в 8 раз быстрее)
- 🎯 **Трекинг оптическим потоком** между детекциями: сорок опорных точек по голове, прямая-обратная проверка, RANSAC — палец или микрофон не искривляют выравнивание
- 🧭 **Калибровка под человека** (минута перед камерой): опорный контур, кроп, устойчивый к мимике, затухание к своему лицу за пределами ваших поворотов
- 👁 **Настоящие моргания и глаза** — модели подмены лишь щурятся; область глаз отдаётся камере при моргании, по желанию — всегда
- 👄 **Маска рта** по чертам лица (губы → основание носа → подбородок), усы не режутся пополам
- 🕒 **Репроекция с низкой задержкой** — последняя готовая подмена сдвигается на самый свежий кадр камеры: тяжёлый улучшатель задерживает мимику, а не положение
- 🧩 Свапперы: Inswapper-128, HyperSwap 1a/1b/1c (256); улучшатели: GPEN-256/512/1024, GFPGAN 1.4, CodeFormer, RestoreFormer++
- 🖥 Адаптивный интерфейс на PySide6 (вкладки Модели / Маска / Движение / Вывод), английский и русский

### Требования

| | |
|---|---|
| ОС | Windows 10/11 (CUDA), macOS Apple Silicon (CoreML) |
| Python | 3.12 |
| GPU | NVIDIA с CUDA 12 для живого режима (проверено на RTX 5060 Ti 16 GB) |
| Модели | скачиваются при первом использовании в `models/` |

### Установка

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
python run.py --execution-provider cuda
```

Для вывода видео нужен `ffmpeg` в `PATH` (или рядом с `run.py`).

### Документация

- [docs/calibration.md](docs/calibration.md) — профили, прокси позы, кроп под мимику, глаза и рот
- [docs/reprojection.md](docs/reprojection.md) — дисплейный трекер и почему он не верит полуприменённой поправке
- [docs/tracking.md](docs/tracking.md) — трекер лица, опорные точки, удержание при потере детекции

<details>
<summary>Структура проекта</summary>

```
modules/
  ui.py                 главное окно PySide6 и живой цикл
  ui_calibration.py     пошаговая калибровка
  calibration.py        профили, прокси позы, стабильные точки
  face_tracker.py       трекер оптическим потоком
  reprojection.py       репроекция на свежий кадр
  face_reveal.py        маски глаз / рта
  face_occluder.py      маска перекрытий XSeg
  cuda_graph.py         сессии с CUDA-графами
  processors/frame/     свапперы и улучшатели
locales/                переводы интерфейса
docs/                   заметки по устройству
```
</details>

### Лицензия

AGPL-3.0, как у исходного проекта. Модели под своими лицензиями; модели подмены — только для некоммерческих исследований.
