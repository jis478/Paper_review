# 1. Introduction

- 그 동안 StyleGAN의 latent space를 활용하여 image manipulation을 하려는 다양한 시도가 있었음.
    - 3DMM in StyleRig, StyleFlow, pSp, Closed-form factorization..
- 하지만 기존 방법론은 사전에 정한 semantic directions을 따라서만 수행하기 때문에 한계가 명확함.
    - 예를 들어 새로운 semantic manipulation을 하고 싶을 때 (예: 안경 추가) 수작업 (annotation, vector 방향 탐색 실험 등)이 필요했음
- CLIP + StyleGAN 을 활용한다면 위와 같은 제약을 벗어날 수 있는 장점이 있음.
- **본 논문은 Image generation이 아닌 Image manipulation을 잘하는 것이 목표임.**
- **pSp 및 Restyle 같은 encoder 논문에 영향을 많이 받음 (동일 저자)**

# 2. Related work

1. **Joint representations between Vision and Language**
    - BERT의 성공으로 최근 transformer를 활용하는 추세
    - CLIP (주어진 text와 image의 semantic 유사성을 예측하는 모델)
2. **Text-guided image generation and manipulation**
    - [AttnGAN](https://github.com/taoxugit/AttnGAN), ManiGAN
    - DALL-E (12-billion params, 24GB GPU memory required for inference)
        - StyleCLIP은 비교적 적은 GPU 리소스를 활용해서 deploy할 수 있는 장점이 있음.
    - TediGAN (concurrent work)
        - CVPR21, 동일하게 StyleGAN을 활용 함. 본 논문과 동일한 (text-image joint learning 컨셉)
    - StyleGAN + CLIP을 먼저 먼저 시도해본 [work](https://towardsdatascience.com/generating-images-from-prompts-using-clip-and-stylegan-1f9ed495ddda) 있음. 하지만 이쪽은 image manipulation이 아닌 generation으로 접근. 본 논문에서는 manipulation으로 차별화 진행

3. **Latent Space Image Manipulation**
- Generator의 Latent space를 활용해서 이미지 manipulation을 시도 (StarGAN)
- StyleGAN latent space (W+) 를 활용해서 이미지 manipulation을 시도
    - Encoder (pSp 등)
    - Explicit annotation (StyleFlow 등)
- 더 나은 Disentanglement를 위해 W+ space를 대체하는 space 활용 시도 ([StyleSpace](https://github.com/betterze/StyleSpace))

# 3. StyleCLIP Text-Driven Manipulation

1. **Latent Optimization**
    - 가장 큰 단점은 수 분의 시간이 걸린다는 점. text-image pair마다 수행해줘야 하는 점.
        - $w_s$ : input image encoding (미리 특정 모델(e4e)로 encoding 시켜 놓음)
        - $w$ :  output encoding (학습 대상 parameter)
        - $t$  :  target text prompt

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ce9bd4ea-4d0e-48d9-8a26-edd7f83b6b6f/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ce9bd4ea-4d0e-48d9-8a26-edd7f83b6b6f/Untitled.png)

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8c7751be-418d-448e-add2-8671ca7e6ef3/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8c7751be-418d-448e-add2-8671ca7e6ef3/Untitled.png)

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/920d9794-2be0-4e3b-85e8-03fd9e66c972/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/920d9794-2be0-4e3b-85e8-03fd9e66c972/Untitled.png)

2. **Latent Mapper**
    - 위에 Optimization 은 너무 시간이 오래 걸림.
    - 기존 pSp 같은 방법론으로 Image를 encoding 잘했으니 유사한 방법을 활용하면 여기서도 잘할 수 있지 않을까? 여기서는 coarse, medium, fine feature를 담당하는 3개의 Mapping network를 통해서 추출. (**StyleGAN에서도 style mixing을 할때 같은 관점에서 수행하는 것에서 따온 듯. 즉, pSp에서 도 유사하게 pyramid feature 방식으로 추출 함**)
    - 3개의 Mapping network는 StyleGAN의 Mapping network와 동일한 구조. 단, StyleGAN에서는 Mapping network가 8개의 FCN 이었지만, 여기서는 각각 Mapping network가 4개의 FCN으로 이루어짐.
    - Target image의 w를 한 번에 생성하는 것이 아닌, 기존 input image의 w에 residual을 더하는 방식으로 생성. ([**ReStyle**](https://arxiv.org/pdf/2104.02699.pdf))
    - 각각의 input image가  W+ 내 다른 point로 encoding 된 후 mapping이 시작 되기 때문에 local mapper
    - **하지만 이 방법은 inference 는 빠르지만 text prompt 마다 학습을 시켜야하는 단점이 있음.**

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9a3d3afb-af72-4e2b-a697-aa6e6766e4bd/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9a3d3afb-af72-4e2b-a697-aa6e6766e4bd/Untitled.png)

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/76b5100a-f685-48f8-8ac2-61747e4c76e1/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/76b5100a-f685-48f8-8ac2-61747e4c76e1/Untitled.png)

    - **Losses**

        CLIP loss로 text에 해당하는 attribute를 반영하고, L2 및 ID loss로 기존 identity 및 다른 부위 attribute를 보존 하는 방향으로 설계.

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/32c18a90-d46b-43fd-a337-900338be5feb/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/32c18a90-d46b-43fd-a337-900338be5feb/Untitled.png)

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/41a4d7b3-accf-44a9-8b14-6917ee31849c/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/41a4d7b3-accf-44a9-8b14-6917ee31849c/Untitled.png)

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/917a1268-0412-42d1-ada6-9ab5561a96d3/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/917a1268-0412-42d1-ada6-9ab5561a96d3/Untitled.png)

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7e02639e-e458-447b-b4eb-84cfe3339b6a/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7e02639e-e458-447b-b4eb-84cfe3339b6a/Untitled.png)

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/96c97815-14bd-4860-afc8-30d73eb47430/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/96c97815-14bd-4860-afc8-30d73eb47430/Untitled.png)

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fec9a6e9-206e-4caf-a981-e25043d6f863/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fec9a6e9-206e-4caf-a981-e25043d6f863/Untitled.png)

**3. Global directions**

- 2번 mapper가 가지고 있는 단점이 있음.

    1) text prompt 마다 학습을 해야하므로 비효율 존재 

    2) local mapper 임. 즉, input image마다 w를 구하고 이를 starting point 시작하기 때문에    
       local mapper로 지칭함. 다른 starting point에서 시작하지만 manipulation direction은
      유사하기 때문에 fine-grained 가 어려움. 

    3) W+ 해서 수행하기 때문에 feature의 disentanglement가 어려움 (W+ 내의 18x512 vector들이 feature 단위로 서로 얽혀있음) 

- 이를 극복하기 위해, W+가 아닌 S space로 encoding을 수행 함.
    - 용어 정리
        - $\Delta s$ : S space 내 변화량
        - $\Delta t$ : CLIP에 input으로 주어지는 text embedding의 변화량
        - $\Delta i$ : CLIP의 joint embedding space 내에서 text manifold I 내 변화량
    - **여기서 궁극적으로 하고 싶은 건 $\Delta t$ 를  S space의 $\Delta s$에 mapping하는 작업. 구하고 싶은 건 $\Delta s$임.**

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/76a96efc-e7f7-4417-a91c-be4b8728c269/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/76a96efc-e7f7-4417-a91c-be4b8728c269/Untitled.png)

    - [manifold  참조](https://www.notion.so/manifold-d00108dfe0a34a7489e581ab963b041c)
    - 즉, 두 개의 이미지 G(s) 와 G(s + @ * $\Delta s$ )가 있는데, 각각 매핑되는 i  와 i  + $\Delta i$ 로 생각해본다면, 실제 두 이미지의 차이는 결국 $\Delta i$로 결정된다고 볼 수 있음.
    - 다시 말하면 $\Delta t$ → $\Delta i$ → $\Delta s$  로 $\Delta s$의 값을 구해볼 수 있음.

    **트릭1) From natural language to $\Delta t$** 

    - CLIP의 text embedding noise를 줄이는 기법 (Radford et al.)
    - 같은 의미를 가진 여러 문장의 embedding을 평균내서 input으로 활용하여 $\Delta t$ 계산에 활용
        - "a bad photo of a {}", "a  cropped photo of the {}" ..
    - 본 논문에서도 이 기법을 활용 함.
        - 단, {} 안에 target attribute + neutral class 가 같이 들어가야 함.
        - 예) a sports car (target: "sports", neutral class: "car")

    2) Channel-wise relevance

    - **다시 생각해보면.. 결국 하고자 하는 것은 주어진 input text 변화 $\Delta t$와 col-linear한 $\Delta i$를 산출해서 S space 내에서 $\Delta s$를 구하는 것임.**
    - 이를 위해서 CLIP의 joint embedding space 내 $\Delta i$ 에 매핑되는 S space 의 각 Style channel의 관련성을 평가해야 함
    - 우선 style code s를 생성 한후, 각 channel에 negative, positive value를 더해서 perturb 시킨다.  G(s) 와 G(s+perturb) Image pairs을 활용해서 CLIP 내에서 $\Delta i_c$ 를 구한 후, style code s의 요소인 channel과  $\Delta i$ 의 관련성을 다음과 같이 산출.

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/290f656f-2540-4c06-9d2c-1805a8a2bf81/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/290f656f-2540-4c06-9d2c-1805a8a2bf81/Untitled.png)

    - 즉, style code의 각 channel이 반영되는 정도를 control 하여 disentanglement를 control할 수 있음.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0f0f6c95-e8dd-4789-abda-2808dae96ac2/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0f0f6c95-e8dd-4789-abda-2808dae96ac2/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6ba4a30e-7c90-402c-905a-87c2fe3ba4fa/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6ba4a30e-7c90-402c-905a-87c2fe3ba4fa/Untitled.png)

- inference에서 target text가 주어지면 "angry". 이걸 기준으로 delta_s를 구하고 싶음.
- clip으로 특정 text에 해당하는 image를 찾을 수 있음.. 그럼 그 이미지는 manifold I에 위치할 거고, 그 정보를 가지고.. S와 비교가 가능함. s 각 채널의 값이 변할때 i에 영향도를 볼 수 있음
- text를 CLIP text encoder로 embedding 시키면 t를 구하는데, delta_t

# 4. Results and Evaluations

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/47b3012a-20ac-4990-8e4f-49427dfbece9/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/47b3012a-20ac-4990-8e4f-49427dfbece9/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/85a10404-1e92-46a6-9215-9f95515d44ec/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/85a10404-1e92-46a6-9215-9f95515d44ec/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b82dc348-d722-45b7-8673-e73f730a04dd/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b82dc348-d722-45b7-8673-e73f730a04dd/Untitled.png)

# 5.  Limitations & Conclusions

- pre-trained 모델을 쓰기 때문에 pre-trained 모델이 제대로 학습하지 못한 이미지,text 에는 성능 안나옴
- dramatic translation 어려움 (예: 호랑이 → 사자 (o), 호랑이 → 늑대 (x))
