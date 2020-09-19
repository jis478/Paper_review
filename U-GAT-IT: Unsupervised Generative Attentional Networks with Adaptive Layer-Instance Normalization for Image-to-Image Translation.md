

U-GAT-IT: 
============================

## Introduction

- 기존에 많은 Image to image translation 논문들이 나왔지만, 대부분은 style transfer와 유사하게 texture를 변환하는 작업에 초점이 맞춰져 있고, shape을 변환시키는 작업에는 약점을 드러 냄 
  (예시: photo2vangogh / cat2dog)
- 본 논문에거는 attention module고 새로운 normalization 방법론을 적용하여 end-to-end로 학습이 가능한 방법론을 제시 함
- 본 논문의 가장 큰 특징은 다음과 같음
   1) 새로 추가되는 aux classifier를 통해 얻어지는 attention map을 활용하여 source와 target 도메인 간 translation에 있어서 중요한 부분과 중요하지 않은 부분을 구분해 집중할 수 있는 장점.
      attention map이 generator와 discriminator에 모두 반영되어 단순히 texture가 아닌 shape translation 까지 이루어질 수 있는 구조 임
      - generator : 이미지 생성 시 두 도메인 간의 차이에 집중 (어떤 부분에 집중해면 더욱 진짜 처럼 생성이 되는지?) 
      - discriminator : 진짜 이미지와 생성 이미지 간의 차이 집중 (어떤 부분에 집중하면 진짜/가짜 구분이 잘 되는지?)
   2) Adaptive Layer Instance Normalization (AdaLIN) 을 도입 함. 이는 기존에 Instance Normalization과 Layer Normalization을 adaptive하게 선택해서 학습하는 방법 임
      AdaLIN 덕분에 모델이 shape과 texture의 변화 정도를 좀 더 쉽게 컨트롤 할 수 있는 역할을 하게 되며, 이로 인해 모델 구조를 바꾸거나 할 필요가 없어짐

~~~
참고로, 본 논문과 유사하게 shape translation을 다른 방법으로 극복하려는 시도가 근래 계속 되고 있음. 기존 CycleGAN를 baseline으로 한 개선 방법 임.

   1) GANHOPPER  [https://arxiv.org/abs/2002.10102] : translation을 한번에 하는게 아니라, 여러번 나눠서 순차적으로(sequentially) 시도하는 방법. 동일한 generator를 여러번 연결하여 25%->50%->75%->100% smooth하게 translation 시도 함
   
   2) ACL-GAN [https://arxiv.org/abs/2003.04858]  : Cycle consistency loss를 변형하여 기존 픽셀단위 loss 계산이 아닌 이미지 간 분포 단위 loss 계산으로 변환하여 좀 더 유연한 translation을 시도 함
~~~

## 2. U-GAT-IT

### 2.1 Model

#### 2.11 Generator

![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/1.jpg)

generator는 encoder $E_s$, $$decoder G_t$$, $$aux classifier  \eta_s(x)$$ 로 이루어져 있으며, 여기서  aux classifier는 input 이미지가 특정 도메인 $$X_s$$ 에서 왔는지에 대한 확률을 표현하며, 다음과 같이 encoder의 각 k-th feature map이 global average pooling 및 global max pooling을 된 후 곱해는 weight $$W^k_s$$를 학습한다.
$$\eta_s(x) = \sigma(\sum_kw^k_s\sigma_(ij)E^{k_(ij)}(x))$$ 

이를 통해서 domain specific attention feature map $$a_s(x) = w_s * E_s(x) = w^k_s * $$  을 구할 수 있으며, 결국 generator $$G_{s->t}$$는 $$G_t(a_s(x))$$가 된다.

한편, generator의 residual block에 AdaLIN을 적용 하기 위해서는 위에서 attention map을 구하기 위해 사용한 fully connected layer로 부터 얻어지는 학습 파라미터 $$\gamma$$ 와 $$\beta$$를 활용하도록 한다.

![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/1.jpg)

$$a_I$$: channel-wise mean
$$u_L$$: layer-wise mean 
$$\sigma_I$$: channel-wise std
$$\sigma_L$$: layer-wise std

* 다양한 normalization 참고자료: [https://www.tensorflow.org/addons/tutorials/layers_normalizations]

* IN, AdaIN 차이점
IN의 경우, BN과 동일하게 input 이미지를 affine parameter ($$\gamma, \beta$$)로 normalize 수행 한다. 많은 DNN케이스에서BN보다 더 빠른 converge를 보였음.

$$ IN(x) = \gamma*((x-\mu(x))/\sigma(x)) + \beta $$
$$\mu_nc(x) = 1/HW * \sum^H_{h=1} \sum^W_{w=1} * x_{nchw} $$
$$AdaIN(x,y) = \sigma(y) * (x-\mu(x)/\sigma(x)) + \mu(y) $$

반면, AdaIN에서는 feature space 상에서 별도의 style 이미지로 부터 얻는 style feature statics (channel-wise mean and variance)를 활용해서 normalization을 수행한다. 즉, affine parameter를 학습하는 것이 아니라 style 이미지로 부터 활용하기 때문에 style transfer의 효과를 볼 수 있다. 

따라서, Style Transfer에서는 AdaIN이 쓰이는데, 각 feature 마다 독립적으로다른 style을 주입해준다는 가정이 있기 때문에 (feature channels이 상호 상관 관계가 없다고 가정) 이로 인해서 style이 아닌 content 정보가 함께 일부 transfer 되는 단점이 존재했다. (예: feature들이 모여서 structure 형성) 하지만 Layer Normalization의 경우 이러한 가정이 없기 때문에 (feature channels에 대해 global statistics 계산) content 정보를 잃을 수 있음.

따라서, AdaIN 및 LN을 adpative하게 적용하는 방법으로 translation 성능을 향상 시킬 수 있었음.

#### 2.1.2 Discriminator

Discriminator는 전반적으로 다른 모델들과 매우 유사하며 주요 차이점은 다음과 같다. 
 
Encoder $$ E_{D_t}$$, Classifier $$C_{D_t}$$, 그리고 Aux classifier $$ \eta_{D_t} $$ 로 이루어져 있으며, 다른 모델들과는 다르게 $$ \eta_{D_t} $$ 와 $$D_t(x)$$ 모두 이미지 $$x$$가 진짜 ($$X_t$$) 인지 가짜 ($$G_{s->t}(X_s)$$)로 부터 온 것인지를 판별하는 역할을 하며, 특히  $$D_t(x)$$는 attention feature maps을 활용하여 $$a_{D_t}(x) = w_{D_t} * E_{D_t}(x)$$ 를 계산 하는데 여기서 $$w_{D_t}$$는 위에[서 언급한 Aux classifier $$ \eta_{D_t} $$로 의해 학습되는 파라미터이며, E_{D_t}(x)는 각 단계에서의 encoding된 feature maps을 의미한다. 

따라서, Discriminator는 $$D_t(x)는 C_{D_t}(a_{D_t}(x))$$가 된다.

## 2.2 Loss function

총 4가지의 Loss function이 쓰이는데,  다른 논문에서 찾아볼 수 있는 Adversarial loss  (LSGAN loss), Cycle loss 및 Identiy loss 외에 추가로 CAM loss를 사용한다. 이는 Generator (어디를 보면 더 진짜 같은 가짜 이미지를 생성이 가능할지?) 와 Discriminator (어디를 보면 domain A / B가 가장 차이가 있는지?) 의 attention 작업을 원할하게 이루어지도록 하는 역할을 하게 된다. 

다시 한번 Aux classifier의 역할을 정리해보면, 

1) Generator의 Aux classifier $$\eta_s(x)$$ : Generator의 encoder가 만들어내는 각각의 feature map에 $$\eta_s(x)$$의 가중치 (attention)을 곱해서 encoding 작업이 더욱 효과적으로 수행 (어떤 feature map이 Generator에 있어서 더 중요한지 판단) 되도록 도와준다.  $$\eta_s(x)$$ 결과 값은 이미지 $$x$$가 domain $$X_s$$에서 올 확률을 의미한다.   
  
2) Discriminator의 Aux classifier $$ \eta_{D_t} $$: 위와 유사하게 $$ \eta_{D_t} $$의 파라미터는 Discriminator의 feature map 중 어떤 것이 중요한지에 대한 가중치 값임. 즉,  결과 값은 input 이미지 $$x$$가 진짜 이미지 ($$X_t$$) 에서 올 확률을 의미한다. 

즉, Generator CAM Loss에서는 $$\eta_s(x)$$는 도메인에 대한 구분을 할 수 있는 loss 반영을 위해 Binary Cross Entropy loss 구성 ($$\eta_s(x)$$가 이미지 $$x$$가 domain $$X_s$$에서 올 확률을 의미하므로)하며,
  
  $$L_{cam}^{s->t}= -(E_{x~X_s}[log(\eta_s(x))] + E_{x~X_t}[log(1-\eta_s(x))])$$
  
Discriminator CAM loss에서는 $$ \eta_{D_t} $$는 도메인이 아닌 진짜/가짜에 대한 구분을 하는 loss 반영을 위해 LSGAN loss 형태로 구성을 한다.

$$L^{D_t}_{cam} = E_{x~X_t}[(\eta_{D_t}(x))^2] + E_{x~X_s}[(1-\eta_{D_t}(G_{s->t}(x))^2]$$
  

