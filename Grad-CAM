1.개요

Grad-CAM(Gradient-weighted Class Activation Mapping)은 특정 target concept (예.강아지로 판별하는 logits 값) 의 gradient를 이용하여 시각화를 수행 함 

여기서 gradient는 마지막 conv layer로 흘러들어가는 gradient로써, Grad-CAM은 target concept의 값을 결정하기 위해 가장 중요한 부분을 시각화 시켜주는 localization map을 생성함

기존 방법론과 큰 차이 점은 Grad-CAM은 다양한 CNN 계열의 알고리즘에 적용가능 하다는 것임

1)	CNN with Fully Connected layers (eg. VGG)
2)	CNNs used for structured outputs (e.g. captioning)
3)	CNNs used in tasks with multi-modal inputs (e.g. VQA) or reinforcement learning, without architectural changes or re-training

추가로 본 논문에서는 Grad-CAM에 high-resolution 시각화 방법을 결합하여 “class-discriminative & high resolution”이 모두 가능한 시각화를 시도했고, 이를 classification, captioning, visual question answering models 등 다양한 분야에 적용해본 결과 그 성능이 우수 함을 보여 줌 

Grad-CAM의 효과를 classification에 관점 정리하면 아래 5가지로 요약 가능 함.

(a)	모델이 실패하는 경우에 대한 Why?에 대한 인사이트 제공  (여기서 “실패”는 모델이 잘못 분류를 한 경우를 의미)
(b)	Adversarial image에 robust 한 시각화 제공 
(c)	Weakly-supervised localization으로 활용될 경우 기존 기법을 능가하는 성능 
(d)	모델이 학습하는 feature에 faithful 한 시각화 제공
(e)	데이터셋의 bias를 발견함으로써 model 성능 일반화에 도움을 줄 수 있음

본 논문에서는 추가로 Grad-CAM이 딥러닝의 예측 결과값에 대한 신뢰성을 제공하고, 딥러닝에 익숙하지 않은 사람들도 성공적으로 “강한”, “약한” 딥러닝 네트워크를 구분할 수 있게 도와준다는 것을 사람을 대상으로 하는 실험으로 확인 함 

1.Introduction
CNN 및 딥러닝 계열의 알고리즘들이 최근 많은 영역에서 성과를 내고 있으나, 해석이 힘든 단점이 존재 함. 예를 들어 AI 시스템이 갑자기 작동하다가 fail해버릴 경우, 그 이유를 알기 힘들어서 대처도 어려운 치명적인 단점이 있음.

즉, Interpretability가 중요함. why they predict what they predict를 이해해야 함

일반적으로 Interpretability와 Accuracy에는 trade-off가 존재 함. 즉, Rule-based 알고리즘은 
Interpretability가 높으나 Accuracy가 낮음. 딥러닝은 여러 layer를 거친 특성의 추상화를 통한 학습과 end-to-end 학습이라는 특성 때문에 특히나 Interpretability가 낮게 됨

기존에 CAM 방법이 제안되었으나, 이는 모델의 fully connected layer를 제거한 상태에서 CNN layer만 가지고 중요한 영역을 판단하기 때문에 위에서 언급한 trade-off (Interpretability가 높아지는 대신 performance가 낮아짐)이 발생함

하지만, 본 논문에서는 모델 아키텍쳐 수정없이, 즉 trade-off 없이 시각화를 수행할 수 있는 Grad-CAM 방법을 제안 함으로써 이런 단점을 극복 함. 

일반적으로 좋은 시각화를 위한 조건 2개가 있음.
1)	Class 분류를 위해 중요한 부위를 하이라이트 하여야 함 (class discriminative)
2)	단순히 하이라이트 하는 것이 아닌, detail한 부분을 잘 표현해야 함 (high resolution)

 


위에 그림에서 볼 수 있듯이, Guided backprop 같은 기법은 high resolution 디테일을 잘 살릴 수가 있으나, class 간의 차이를 나타내는 것은 어렵다. (b) 와 (h) 비교해면 알 수 있음.

반대로 Grad-CAM 같은 경우는 class 간의 차이를 잘 보여주는 장점이 있다. (c)와 (i) 참조

따라서 Guided backprop같은 pixel-space gradient 시각화를 Grad-CAM과 결합시켜서 high resolution과 class discriminative를 둘다 달성할 수 있는 “Guided Grad-CAM”을 만들어 낼 수 있다. (d) 와 (j) 참조 할 것


2. Related Work
본 논문에서는 아래 3개 영역에 대한 기존 방법들을 참고하고 있음.

Visualizing CNNs
이전에도 시각화를 위한 다양한 방법들이 존재했음
pixel intensities에 대해 class score를 편미분함으로써 class에 영향을 미치는 pixel 분석에 대한 접근 방법으로 Deconv, Guided backprop처럼 기존 gradient에  변형을 가하는 방법이 있었으나, 이는 class-discriminative하지 않고, high resolution 시각화에 포커스를 둔 연구 결과임 

한편, 네트워크의 unit의 activation 값을 max로 만드는 이미지 생성하는 방법도 있으나, 이는 모델을 전반적으로 시각화할뿐, 특정이미지에 대한 시각화 시도는 아님

Assessing Model Trust
기존에 모델의 신뢰성을 확인할 수 있는 연구가 있었으며, 이에 연장선상에서 Grad-CAM이 자동화된 시스템에서 쓰일 경우 사용자가 얼마나 모델에 대한 신뢰감을 가질 수 있는 지 그 정도를 평가 함  

Weakly supervised localization
Weakly supervised localization 이란 class label 정보만 가지고 이미지 내에서 특정 object가 위치하는 곳을 알 수 있는 방법임. 본 논문과 가장 밀접한 이전 연구 결과는 CAM인데, CAM은 Fully-connected layer를 conv 및 Pooling layer로 대체하는 단점이 존재 함 (CAM은 특정한 구조를 가진 네트워크에만 적용이 가능. conv feature maps → global average pooling → softmax layer) 따라서 (Fully-connected layer가 없기 때문에) 모델의 성능이 떨어질 수 밖에 없음. 

본 논문에서 제안하는, Grad-CAM은 CAM의 일반화된 버전이라고 생각할 수 있음.

다른 방법으로는 perturbations등의 여러 방법이 있으나 본 논문에서 제안하는 Grad-CAM이 성능/속도 면에서 훨씬 우수한 결과를 보임 

3. Approach
Feature의 위치 정보는 fully connected layer를 거치면 사라지게 되기 때문에 가장 마지막 conv layer가 가장 높은 수준의 semantics과 디테일한 위치 정보에 대한 타협점이라고 볼 수 있음. 마지막 conv layer의 뉴런들은 이미지 내에서 semantic class-specific 정보 존재 여부를 찾게 됨

Grad-CAM은 마지막 conv layer로 흘러들어가는 gradient 정보를 활용해서 각 뉴런이 classification을 위해 가장 중요시 하는 정보를 이해하게 됨

●	수식 분석
예를 들어 class c에 대한  softmax layer를 통과하기 전의 logits 값을  라고 하고,  
 가 input 이미지에 대한 영향을 분석하고 싶은 conv layer (일반적으로 마지막 layer)의 k 번째 feature map라고 정의한다. 그러면 gradient 값을 가지고 다음과 같이  global average pooling을 수행하여  뉴런의 importance weights 를 다음과 같이 계산할 수 있다. (i,j 는 k 번째 feature map 내 픽셀의 위치) 


 

즉,   는 특정 class c에 대해 특정 conv layer의 feature map k가 미치는 영향도를 의미한다.

각 feature map k의 영향도에 feature map k를 곱해주는 weighted combination을 통해 다음과 같이 class c에 대한 Grad-CAM을 구할 수 있다. 
 
1)	Grad-CAM은 k개의 feature map을 linear combination 한 것임 (사이즈는 feature map 사이즈임)
2)	Relu 적용을 통해 positive 영향이 있는 feature만 시각화 함 
-> 즉, linear combination으로 생성된 heat-map 이미지의 특정 pixel 값이 (-)로 되어있다면 이는 class c이 중요시하는 pixel이 아닌 것을 의미한다. (다시 말하면 class c의 logit 값이 높아지면 pixel의 값도 높아져야 하는데 반대의 경우로, 이는 다른 class에 속한 pixel일 가능성이 높음) 
 
  



Grad-CAM as a generalization to CAM
Grad-CAM은 CAM의 일반화된 버전으로 볼 수 있음. CAM은 위에서 언급한대로 특정 구조 (i.e. conv feature maps → global average pooling → softmax layer) 를 가진 경우에만 적용할 수 있기 때문임. 아래 수식에서 각 feature map에 대해서 global average pooling을 거친 후 class feature weights를 곱해주는데, 이 weights는 재학습을 통해서만 구할 수 있는 단점이 있음

  
     
  
즉, Grad-CAM에서는 이 weight를 영향도 로  대체하여 일반화 시킨 것임.

Grad-CAM의 큰 장점 중에 하나는   값이 반드시 logit 값일 필요가 없고, 미분가능한 함수면 되기 때문에 classification, captioning 및 Q&A등 다양한 분야에서 쓰일 수가 있다는 것임.


Guided Grad-CAM
Grad-CAM은 feature map 사이즈를 기준으로 러프한 heat-map을 생성하기 때문에 원본 이미지 pixel 단위의 정교한 highlight를 할 수는 없음 (high-resolution 문제). 이를 극복하기 위해 Grad-CAM + Guide Backprop를 적용하여 class-discriminative 하고 high-resolution이 되는 시각화를 달성 함.


4. Evaluating Localization 
Weakly-supervised Localization
ImageNet 데이터 내 존재하는 object들에 대해서 Grad-CAM으로 heat map을 생성 후, 생성된 heat map 내에서 max intensity 15% 이상의 상호 연결된 pixel을 object로 가정하고 bounding box를 쳤음. 즉, bounding box의 error를 계산해본 결과, 다른 heat map 생성 기법보다 월등하게 좋은 성능을 보임. 특히, CAM의 경우에는 VGG16을 변형해서 돌려야 하기 때문에 heat map을 생성하더라도 classification 성능이 낮아지는 단점을 재확인 함.

 


5. Evaluating Visualizations 
사용자가 Grad-CAM의 결과에 대해 신뢰감을 가질 수 있는지에 대해  43명의 사람을 대상으로 실험을 진행했음.  

1)	Guided Grad-CAM이 다른 시각화기법보다 더 Class-discriminative 한 시각화 제공
2)	Guided Grad-CAM을 적용하게 되면 성능이 좋은 모델(VGG16)과 나쁜모델 (AlxeNet)을 판단할 수 있는 근거를 시각적으로 제공 할 수 있음 (즉, 동일한 특정 이미지에 대해 두 다른 모델 (VGG16, AlexNet)에 대한 Grad-CAM을 수행했을때, Grad-CAM 결과가 더 좋은 모델이 ImageNet 기준 Accuracy도 더 높은 모델임 -> Grad-CAM이 신뢰성 제공

 


한편, 시각화에서는 faithfulness도 중요한 요소인데, faithfulness는 모델을 통해 배운 함수을 얼마나 잘 설명하느냐에 대한 문제임.

이를 실험하기 위해 Image occlusion 기법을 써봤음. 이는 이미지의 일부를 masking하여 CNN score가 어떻게 변하는지를 보는 것인데, 흥미롭게도 CNN score를 변하게 하는 masking 영역은 Grad-CAM의 heat map이 높은 값으로 부여하는 영역인것을 발견 하였으며 (corr 0.254) 다른 기법 보다 더욱 뛰어난 성능임을 확인 함.즉, Grad-CAM은 원본 모델 (eg. VGG16)이 가지고 있는 함수 표현 능력에 충실한 시각화임을 확인 할 수 있음.

6. Diagnosing image classification CNNs

모델이 잘못 분류한 이미지에 대한 설명을 Grad-CAM을 통해 설명할 수 있음. 즉, 
Grad-CAM을 통해서 모델이 왜 잘못 예측했는지에 (어떤 영역을 잘못 본것인지) 대해 설명이 가능 함. 
 


Adversarial 기법을 이용하여 모델이 잘못된 class로 예측하도록 이미지를 생성한 후, 가짜 이미지에 Grad-CAM을 적용해도 Grad-CAM이 원래 정답 object를 잘 하이라이트 하는 것을 볼수 있음. 즉, 이미지에 Adversarial noise가 들어있어서 모델이 최종 판정을 잘못하더라도, Grad-CAM의 시각화는 robust한 성능을 보이는 것을 확인함. 

 

Dataset에 bias가 있으면 모델이 generalize을 잘못하는 경우가 존재함. 즉, ImageNet pre-trained모델을 가져와서 250장의 의사 vs 간호사 학습이미지로 모델을 훈련시킬 경우, validation set에 대해서는 높은 성능을 보이지만 test set에서는 낮은 성능(82%) 을 보이는 경우가 있음.

이를 Grad-CAM으로 확인 결과, 이미지의 헤어스타일 및 얼굴 영역을 하이라이트 되는 등 모델이 gender stereotype을 배운 것 확인 하였고, 따라서 학습 이미지를 검사해보니 의사의 대부분이 남자, 간호사의 대부분이 여자여서 그런것으로 확인 됨.따라서 학습이미지 수량을 동일하게 한 상태에서 여자 의사 이미지 및 남자 간호사 이미지 비율을 조정해서 재학습 한 결과 test set에서도 좋은 성능을 보여줌 (90%) 

 



7. Counterfactual Explanations
마지막으로, 기존 Grad-CAM에 다음과 같은 추가하였음. 원래 Grad-CAM은 모델이 특정 class c로 판별하는데 가장 중요한 영역을 하이라이트 했다면, Counterfactual explanations를 적용하면 모델이 class c가 아닌 다른 class로 판별하도록 하는데 있어서 가장 중요한 영역을 하이라이트 하는 기능임. 

따라서, Counterfactual explanations가 하이라이트 하는 영역의 object가 지워지면 모델은 class c에 대해 더욱 확신을 가지고 판별 할 수가 있는 것임. 

 
 


9. Conclusion 

Grad-CAM은 기존의 CAM이 가지고 있는 한계 (재학습 필요, 특정 네트워크 구조가 필요)를 극복하여 어떤 CNN 네트워크에도 적용 가능한 장점이 있음. 

Grad-CAM을 high-resolution 시각화 기법과 함게 적용 (eg. Guided Grad-CAM) class discriminative 뿐만 아니라 high resolution 까지 함께 시각화 할 수 있는 장점이 있음

한편, 실제 실험을 통해 사용자가 Grad-CAM을 통해 모델의 trust 및 faithfulness를 확인할 수 있고, 학습이미지의 bias까지 적발 할 수 있음을 확인 하였음.

마지막으로 Grad-CAM은 일반 CNN 모델 뿐만 아니라 captioning, VQA등 다양한 CNN 결합 모델에 대해서도 시각화를 위해 쓰일 수 있음을 확인 함 

Future work로는 CNN 뿐만 아니라 강화학습, NLP 등에 대한 비-CNN 계열의 모델의 시각화에 대한 연구가 진행되었으면 함.



 

