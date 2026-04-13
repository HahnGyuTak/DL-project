# XAI506 DeepLearning Project

- EfficientSAM 세그멘테이션
- Grounding DINO Open-vocabulary Detection
- LLaVA VQA
- Segmentation + SD3 Inpainting

> [Web Demo Guidelines](docs/README_WEB_DEMO.md)

---

## 1. Set Up

### Requirements

- Python 3.10+ , `pip`
- NVIDIA GPU + CUDA
> (선택) Cloudflare Pages 배포 시 `node`, `npm`, `wrangler`


### Git clone

```bash
git clone https://github.com/HahnGyuTak/DL-project.git
cd DL-project
```

### Python Environment

#### `venv` 

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_api.txt
```

#### `conda` 

```bash
conda create -n dl-project python=3.10 -y
conda activate dl-project
pip install --upgrade pip
pip install -r requirements_api.txt
```

---

## 2. Explanation & I/O

**Input image**

![Example](img/xai506_example_image.jpg)


### 2.1 Efficient SAM 

[code](ipynb/efficientsam_segmentation.ipynb)

**Model:** `merve/EfficientSAM`

EfficientSAM은 SAM 계열의 경량 segmentation 모델이다. 노트북에서는 입력 이미지 `img/xai506_example_image.jpg`를 불러온 뒤, point prompt 또는 box prompt를 모델 입력에 맞게 resize한다. 
point prompt는 foreground/background label을 사용하고, box prompt는 두 꼭짓점 좌표로 영역을 지정한다.

**Output:** 선택된 mask의 IoU 후보 중 가장 높은 mask를 선택하고, 원본 이미지 위에 segmentation 영역을 overlay한 결과를 출력한다.

![SAM output](img/output_sam.webp)


### 2.2 Open vocab detection

[code](ipynb/grounding_dino_open_vocab_detection.ipynb)

**Model:** `IDEA-Research/grounding-dino-tiny`

Grounding DINO는 사용자가 지정한 텍스트 label을 기반으로 객체를 검출하는 open-vocabulary detection 모델이다. 노트북에서는 `candidate_labels`를 `person`, `table`, `cup`처럼 자유롭게 지정하고, label을 소문자 + 마침표 형태의 text prompt로 변환한 뒤 이미지와 함께 모델에 입력한다.

**Output:** 텍스트 조건과 맞는 객체의 bounding box, confidence score, label을 계산하고, 원본 이미지 위에 박스와 label을 그려서 시각화한다.

`candidate_labels = ['person']`

![DINO output](img/output_dino.png)

`candidate_labels = ['table', 'cup']`

![DINO output](img/output_dino2.png)


### 2.3 LLava VQA

[code](ipynb/llava_vqa_xai506_example.ipynb)

**Model:** `llava-hf/llava-1.5-7b-hf`

LLaVA는 이미지와 텍스트 질문을 함께 입력받아 답변을 생성하는 vision-language 모델이다. 
이미지 tensor와 질문 token을 함께 모델에 넣어 답변을 생성한다.

**Output:** 질문에 대한 자연어 답변을 출력한다. GPU 환경에서는 `float16`으로 로드해 추론한다.

`question = 'What is happening in this image? Please answer in Korean.'`


```
Model: llava-hf/llava-1.5-7b-hf
Device: cuda
DType: float16
--- Question ---
What is happening in this image? Please answer in Korean.
--- Answer ---
이 이미지에서는 학생들이 학교에서 수업을 듣고 있는 모습을 볼 수 있습니다. 학생들은 컴퓨터와 라피를 사용하며, 책과 커피잔을 볼 수 있습니다. 이 모습은 학생들이 집중하고 있는 학습 환경을 보여줍니다.
```
