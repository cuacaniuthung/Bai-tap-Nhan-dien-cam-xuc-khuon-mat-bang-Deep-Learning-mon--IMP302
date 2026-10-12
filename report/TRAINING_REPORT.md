# Bao Cao Ket Qua Training - Emotion Detection (FER-2013)

**Ngay training:** 02/10/2026
**Moi truong:** Kaggle Notebook - GPU NVIDIA Tesla T4
**Dataset:** FER-2013 (/kaggle/input/datasets/msambare/fer2013)
**Framework:** PyTorch
**Thoi gian chay tong:** ~68 phut (4098 giay)

---

## 1. Tong Quan Bai Toan

### Dataset FER-2013
- **7 lop cam xuc:** angry, disgust, fear, happy, neutral, sad, surprise
- **Tap train:** ~28,709 anh
- **Tap test:** 7,178 anh
- **Kich thuoc anh:** 48x48 pixels, grayscale

### Dac diem quan trong cua dataset
- **Class imbalance nang:** `happy` chiem ti le cao nhat, `disgust` cuc ky it
- **Noisy labels:** Ngay ca con nguoi gan nhan cung chi dong thuan ~65-70%
- **Anh nho, chat luong thap:** 48x48 pixels grayscale gay kho khan cho transfer learning tu ImageNet (von duoc train tren anh mau RGB, kich thuoc lon hon)

---

## 2. Kien Truc Hai Model

### Model 1 - Custom CNN (M1)
- Kien truc CNN xay dung tu dau (from scratch)
- Nhieu lop Conv2D -> BatchNorm -> ReLU -> MaxPool
- Classifier head voi Dropout de tranh overfitting
- Toi uu hoa: Adam (lr=1e-3, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)

### Model 2 - MobileNetV2 (M2) - Transfer Learning
- Pretrained tren ImageNet
- **Phase 1 (10 epochs):** Freeze toan bo backbone, chi train classifier head
- **Phase 2 (20 epochs):** Unfreeze toan bo backbone, fine-tune toan bo model
- Toi uu hoa Phase 1: Adam (lr=1e-3)
- Toi uu hoa Phase 2: Adam (lr=1e-4, weight_decay=1e-4) + CosineAnnealingLR

### Xu ly Class Imbalance
- Ca 2 model deu dung CrossEntropyLoss(weight=class_weights) voi trong so duoc tinh tu phan phoi lop

---

## 3. Ket Qua Training

### 3.1. Custom CNN - 30 Epochs

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1     | 1.9654    | 1.9207   | 17.33%    | 27.07%  |
| 5     | 1.8935    | 1.8382   | 26.16%    | 32.11%  |
| 10    | 1.8432    | 1.7806   | 28.67%    | 32.72%  |
| 15    | 1.8030    | 1.7547   | 28.95%    | 34.37%  |
| 20    | 1.7508    | 1.6593   | 23.18%    | 35.59%  |
| 25    | 1.5573    | 1.4047   | 38.45%    | 45.29%  |
| 30    | 1.4071    | 1.2447   | 45.69%    | 52.69%  |

**Best Val Accuracy: 52.69%**

#### Nhan xet CNN:
- Model hoc rat cham trong 20 epoch dau (val_acc chi quanh 32-38%)
- Buoc nhay dang ke tu epoch 20 den 30 (val_acc tu 35% len 52%)
- Dac biet: val_acc > train_acc o nhieu epoch (vi du epoch 21: train 26.34% vs val 41.60%)
  -> Model dang **underfitting** nang, capacity cua CNN chua du de hoc cac dac trung phuc tap
- Loss van dang giam o epoch 30 -> Co the train them epochs, nhung kien truc qua don gian

---

### 3.2. MobileNetV2 - Phase 1: Classifier Only (10 Epochs)

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1     | 1.7848    | 1.7951   | 29.49%    | 26.76%  |
| 5     | 1.6697    | 1.7242   | 33.93%    | 30.87%  |
| 10    | 1.6301    | 1.6424   | 35.78%    | 34.98%  |

**Best Val Accuracy Phase 1: 37.36%**

#### Nhan xet Phase 1:
- Chi train classifier voi backbone frozen -> ket qua khiem ton (37.36%)
- Ly do: MobileNetV2 pretrain tren ImageNet (anh mau RGB, 224x224), trong khi FER-2013 la
  anh grayscale nho 48x48 -> features tu backbone khong phu hop hoan toan voi domain moi
- Val_acc dao dong, khong on dinh -> backbone chua duoc fine-tune de hoc dac trung cua bieu cam

---

### 3.3. MobileNetV2 - Phase 2: Full Fine-tuning (20 Epochs)

| Epoch  | Train Loss | Val Loss | Train Acc | Val Acc                 |
|--------|-----------|----------|-----------|-------------------------|
| 1      | 1.3932    | 1.2507   | 47.69%    | 52.24%                  |
| 2      | 1.1640    | 1.0891   | 56.44%    | 58.41%                  |
| 3      | 1.0475    | 0.9988   | 60.25%    | 62.25%                  |
| 5      | 0.8997    | 0.9658   | 65.01%    | 64.11%                  |
| 8      | 0.7681    | 0.9323   | 69.37%    | 66.58%                  |
| 10     | 0.7004    | 0.9255   | 72.36%    | 67.69%  <- BEST         |
| 12     | 0.6257    | 0.9739   | 75.08%    | 66.66%                  |
| 15     | 0.5290    | 1.0017   | 78.93%    | 66.80%                  |
| 18     | 0.4340    | 1.0826   | 82.87%    | 67.04%                  |
| 20     | 0.4078    | 1.1721   | 84.09%    | 67.07%                  |

**Best Val Accuracy Phase 2: 67.69% (Epoch 10)**

#### Nhan xet Phase 2:
- Unfreeze toan bo backbone tao ra buoc nhay vot ngay lap tuc: tu 37.36% len 52.24% chi sau 1 epoch
- **Overfitting ro rang bat dau tu epoch 11:**
  - Train acc tiep tuc tang: 73.29% -> 84.09%
  - Val acc **giam va dao dong**: 66.62% -> 67.07% (khong cai thien)
  - **Khoang cach Train-Val cuoi Phase 2: ~17%** -> dau hieu overfitting nghiem trong
- Val loss tang tu epoch 10 tro di (0.9255 -> 1.1721) trong khi train loss van giam

---

### 3.4. Ket Qua Cuoi Cung tren Test Set

| Model          | Best Val Acc | Test Accuracy | Anh sai / Tong             |
|----------------|-------------|---------------|----------------------------|
| Custom CNN     | 52.69%      | ~52%          | ~3,438 / 7,178             |
| MobileNetV2    | 67.69%      | 67.7%         | 2,319 / 7,178 (32.3%)      |

**Model tot nhat: MobileNetV2 voi 67.7% accuracy tren test set**

---

## 4. Phan Tich Van De

### 4.1. Overfitting trong MobileNetV2 Phase 2

**Bieu hien:**

```
Epoch 10: Train 72.36% vs Val 67.69%  (khoang cach: 4.67%)  <- Diem tot nhat
Epoch 20: Train 84.09% vs Val 67.07%  (khoang cach: 17.02%) <- Overfitting
```

**Nguyen nhan:**

1. **Learning rate Phase 2 (1e-4) van con kha lon** khi unfreeze toan bo backbone
   -> model "pha" cac features da duoc pretrain, hoc lai theo huong overfit training data
2. **Khong du data augmentation** -> model memorize training samples thay vi hoc patterns tong quat
3. **FER-2013 co noisy labels** -> model co hoc ca cac nhan nhieu -> overfitting
4. **So epochs Phase 2 qua nhieu:** Best val acc dat o epoch 10, nhung van train tiep den epoch 20

### 4.2. Custom CNN Underfitting

**Bieu hien:**

```
Epoch 21: Train 26.34% vs Val 41.60%  -> Train thap hon Val (bat thuong)
Epoch 30: Train 45.69% vs Val 52.69%  -> Model chua hoc du tu training data
```

**Nguyen nhan:**

1. **Kien truc CNN qua don gian** -> capacity khong du de hoc bieu dien phuc tap cua bieu cam
2. **Anh 48x48 grayscale** -> it thong tin de hoc
3. Class_weights co the lam thay doi phan phoi gradient khi training

---

## 5. De Xuat Cai Thien va Ly Do

### 5.1. Giai Quyet Overfitting (Uu tien cao nhat)

#### a) Early Stopping chat che hon `[DA AP DUNG]`
- **Van de hien tai:** Model da dat best val acc tai epoch 10 (Phase 2) nhung van train 10 epoch nua
  -> lang phi tai nguyen, tang nguy co overfitting
- **Giai phap:** Dung Early Stopping voi patience=5, dung khi val_acc khong cai thien sau 5 epoch
- **Ket qua ky vong:** Khong nhat thiet cai thien accuracy nhung tranh them van de va tiet kiem thoi gian

#### b) Tang cuong Dropout va Weight Decay `[MOT PHAN - xem Muc 9]`
- **Van de hien tai:** Regularization hien tai chua du manh de kiem che overfitting (20 epoch Phase 2)
- **Giai phap:** Tang dropout rate trong classifier head (tu ~0.2 len 0.4-0.5), tang weight_decay (tu 1e-4 len 5e-4)
- **Ly do:** Dropout manh hon ngan model phu thuoc qua vao mot so neuron cu the.
  Weight decay phat cac trong so qua lon, buoc model hoc bieu dien tong quat hon

#### c) Data Augmentation manh hon va dung cach `[DA AP DUNG]`
- **Van de hien tai:** FER-2013 la dataset nho (~28K anh), de overfit neu khong augment du
- **Giai phap:** Ap dung RandomHorizontalFlip, RandomRotation (+-10 do), RandomBrightness,
  RandomContrast, RandomCrop
- **Luu y:** Khong augment qua manh vi anh 48x48 rat nho (khong dung vertical flip - khuon mat lon nguoc
  khong tu nhien, khong dung rotation lon - bien dang anh qua nhieu)

### 5.2. Cai Thien Training Strategy

#### d) Gradual Unfreeze thay vi Unfreeze toan bo `[DA AP DUNG - hieu qua ro ret]`
- **Van de hien tai:** Phase 2 unfreeze toan bo backbone cung mot luc voi lr=1e-4 -> buoc nhay qua lon,
  de "pha" features da pretrain tot
- **Giai phap:** Unfreeze tung block tu tren xuong (layer cuoi -> layer dau), moi giai doan 5 epochs
  voi lr nho dan:
  - Giai doan 1: Chi unfreeze classifier + block 18 (lr=1e-4)
  - Giai doan 2: Them block 15-17 (lr=5e-5)
  - Giai doan 3: Them block 10-14 (lr=1e-5)
- **Ly do:** Cac lop dau cua MobileNetV2 hoc edges/textures co ban (it can thay doi voi domain moi),
  cac lop cuoi hoc high-level features (can fine-tune nhieu hon). Unfreeze dan giu lai knowledge
  tu ImageNet va hoc domain moi mot cach muot ma, tranh "catastrophic forgetting"

#### e) Learning Rate nho hon cho Phase 2 `[DA AP DUNG]`
- **Van de hien tai:** lr=1e-4 cho Phase 2 (full unfreeze) la kha lon -> cac layer dau cua backbone
  bi anh huong qua nhieu
- **Giai phap:** Dung lr=1e-5 hoac lr=5e-6 cho Phase 2, ket hop voi warmup (tang dan trong 2-3 epoch dau)
- **Ly do:** Learning rate nho hon cho phep fine-tune tinh te ma khong pha vo cau truc
  da hoc tu ImageNet pretraining

### 5.3. Cai Thien Loss Function

#### f) Label Smoothing `[DA AP DUNG]`
- **Van de hien tai:** FER-2013 noi tieng voi noisy labels - nhieu anh duoc gan nhan khong chac chan
  (vi du: bieu cam sad vs disgust rat kho phan biet). CrossEntropyLoss chuan yeu cau model
  "tu tin 100%" vao tung nhan
- **Giai phap:** Dung label_smoothing=0.1 - "lam mem" one-hot labels tu [1.0, 0, 0, ...]
  thanh [0.9, 0.0143, 0.0143, ...]
- **Ly do:** Khi nhan goc co nhieu, model khong nen bi phat nang vi thieu tu tin.
  Label smoothing giup model khong overconfident, giam overfitting dac biet khi labels co nhieu

#### g) Focal Loss `[CHUA AP DUNG]`
- **Van de hien tai:** Class imbalance (disgust <<< happy), du da co class_weights nhung
  CrossEntropyLoss van de bi dominated boi cac class pho bien
- **Giai phap:** Focal Loss tu dong giam trong so cua cac "easy samples" (nhung anh model da hoc tot)
  va tap trung vao cac "hard samples"
- **Ket qua ky vong:** Model hoc tot hon tren cac class kho va hiem (disgust, fear)

### 5.4. Cai Thien Kien Truc (Dai han)

#### h) Doi sang EfficientNetB0 hoac EfficientNetB2 `[DA AP DUNG - dung EfficientNetB0]`
- **Ly do:** EfficientNet hieu qua hon MobileNetV2 tren cac task image classification nho, duoc thiet ke
  voi compound scaling (width x depth x resolution cung luc) -> accuracy cao hon voi cung so parameters.
  EfficientNetB0 cho thay cai thien ~2-3% tren nhieu benchmark so voi MobileNetV2

#### i) Them Attention Mechanism (CBAM - Channel & Spatial Attention) `[DA AP DUNG]`
- **Ly do:** Bieu cam khuon mat tap trung vao cac vung cu the (mat, mieng, long may) - khong phai toan bo
  anh deu quan trong nhu nhau. CBAM giup model tu hoc "nhin vao dau" thay vi xu ly toan bo anh dong deu,
  cai thien dang ke kha nang phan biet bieu cam tinh te

---

## 6. Ket Qua Ky Vong sau Cai Thien

| Cai tien ap dung                               | Val Accuracy ky vong | Thuc te dat duoc (xem Muc 9) |
|------------------------------------------------|----------------------|-------------------------------|
| Hien tai (baseline)                            | 67.7%                | 67.7% (MobileNetV2, bao cao goc) |
| + Early Stopping + Dropout manh hon            | ~68%                 | - |
| + Data Augmentation dung cach                  | ~69-70%              | - |
| + Gradual Unfreeze + LR nho hon                | ~70-72%              | - |
| + Label Smoothing                              | ~72-73%              | - |
| + Focal Loss                                   | ~72-74%              | Chua ap dung |
| + EfficientNet + CBAM Attention                | ~74-76%              | **70.40%** (M2 solo) |
| + Ensemble M1+M2 (ky thuat moi, ngoai du kien ban dau) | - | **70.77%** |

> **Nhan xet:** Ket qua thuc te (70.40%) thap hon muc ky vong cao nhat (~74-76%) nhung van cai thien
> dang ke so voi baseline (+2.7 diem %, tu 67.7% len 70.40%). Chi tiet nguyen nhan va qua trinh thu nghiem
> xem Muc 9.

> **Luu y thuc te:** State-of-the-art tren FER-2013 (~75-76%) dat duoc bang cac kien truc rat phuc tap,
> ensemble nhieu models, va ky thuat tien tien.
> **67.7% la ket qua tot va hop ly cho bai tap hoc thuat voi 2 model don gian.**

---

## 7. Files Output (Baseline)

| File                           | Kich thuoc | Mo ta                                     |
|--------------------------------|-----------|-------------------------------------------|
| model_cnn.pt                   | 5.30 MB   | Weights cua Custom CNN                    |
| model_mobilenet.pt             | 10.47 MB  | Weights cua MobileNetV2                   |
| class_distribution.png         | 0.07 MB   | Bieu do phan phoi class trong dataset     |
| confusion_matrix.png           | 0.17 MB   | Ma tran nham lan cua ca 2 model           |
| training_curves_cnn.png        | 0.11 MB   | Do thi loss/accuracy cua CNN              |
| training_curves_mobilenet.png  | 0.10 MB   | Do thi loss/accuracy cua MobileNetV2      |
| wrong_predictions.png          | 0.16 MB   | 15 anh du doan sai cua MobileNetV2        |

---

## 8. Ket Luan (Baseline Ban Dau)

Ket qua training cho thay **MobileNetV2 voi progressive fine-tuning vuot troi hon han Custom CNN**
(67.7% vs 52.7%). Tuy nhien, ca 2 model deu co van de ro rang can giai quyet:

- **Custom CNN:** Underfitting do kien truc qua don gian, khong du capacity cho bai toan phuc tap
- **MobileNetV2:** Overfitting tu epoch 11 cua Phase 2 tro di (train acc tang manh nhung val acc
  khong cai thien)

Nguyen nhan chinh cua overfitting:
1. Learning rate Phase 2 con cao, unfreeze backbone qua dot ngot
2. Regularization (Dropout, Weight Decay) chua du manh
3. FER-2013 la dataset nho voi noisy labels - rat de overfit

**Huong uu tien cai thien (theo muc do tac dong/cong suc):**

1. **Gradual Unfreeze** - Thay doi lon nhat ve training strategy, co the tang 3-4% accuracy
2. **Label Smoothing** - Thay doi nho nhung hieu qua voi noisy labels cua FER-2013
3. **Early Stopping** - De implement, tranh lang phi va overfitting them

---

## 9. Cap Nhat: Ket Qua Sau Khi Ap Dung Cac Cai Tien

**Ngay cap nhat:** 05/10/2026

Sau bao cao baseline o tren, cac de xuat o Muc 5 da duoc trien khai va kiem chung thuc te qua nhieu
lan training. Phan nay ghi lai nhung gi da thuc su thay doi va ket qua cuoi cung.

### 9.1. Thay doi kien truc va ky thuat da ap dung

| Ky thuat | Chi tiet trien khai |
|---|---|
| Backbone M2 | Doi tu **MobileNetV2** sang **EfficientNet-B0** |
| Attention | Them **CBAM** (Channel + Spatial Attention) sau backbone.features, truoc classifier |
| Gradual Unfreeze | Chia lai thanh **3 phase** thay vi 2: (1) freeze toan bo backbone - 10 epoch, (2) unfreeze block 6-8 - 15 epoch, (3) unfreeze toan bo - 20 epoch |
| Learning rate | Phase 3 giam con **5e-5** (thay vi 1e-4 o baseline) + **CosineAnnealingLR** |
| Weight decay | Tune qua nhieu muc (5e-4 -> 1.5e-3 -> **1e-3**), 1.5e-3 gay giam diem nhe nen chon lai 1e-3 |
| Label Smoothing | Ap dung trong CrossEntropyLoss ket hop class_weights |
| Data Augmentation | RandomHorizontalFlip, RandomRotation(10 do), ColorJitter; **RandomErasing** giu lai cho M1, bo khoi M2 sau khi xac nhan lam giam diem |
| Mixup | Ap dung o Phase 2 va 3 cua M2 (alpha=0.2); sua lai cach tinh train accuracy theo trong so `lam` de do thi phan anh dung thuc te (truoc do train acc bi danh gia thap gia tao do nhan bi tron) |
| Early Stopping | patience rieng cho tung phase (4-7 tuy phase) |
| Hard Negative Mining (OHEM) | **Da thu nghiem o Phase 3** (chi lay top 70% sample loss cao nhat moi batch de backward) - **khong cai thien ket qua**, dong thoi lam tang overfitting (train-val gap len den 7.4%). Da **loai bo** khoi cau hinh cuoi cung |
| Ensemble | **Ky thuat moi, ngoai du kien ban dau** - ket hop xac suat softmax cua M1 + M2 theo trong so, khong can train lai |

### 9.2. Ket qua cuoi cung tren Test Set

| Model | Test Accuracy | TTA | Macro F1 | Params | FPS (GPU) |
|---|---|---|---|---|---|
| M1 - Custom CNN | 64.85% | 65.51% | 0.6227 | 1,321,191 | 960.6 |
| M2 - EfficientNet-B0 + CBAM | 70.40% | 70.20% | 0.6893 | 4,542,693 | 557.1 |
| **Ensemble (M1 x0.2 + M2 x0.8)** | **70.77%** | - | **0.6938** | - | - |

So voi baseline MobileNetV2 (67.69%), M2 moi (EfficientNet-B0+CBAM) dat **70.40%** (+2.71 diem %),
va **Ensemble dat 70.77%** - ket qua cao nhat trong toan bo qua trinh thu nghiem.

### 9.3. Classification Report - M2 (EfficientNet-B0 + CBAM)

```
              precision    recall  f1-score   support

       Angry       0.61      0.64      0.63       958
     Disgust       0.70      0.70      0.70       111
        Fear       0.61      0.51      0.55      1024
       Happy       0.89      0.88      0.89      1774
     Neutral       0.63      0.71      0.67      1233
         Sad       0.60      0.56      0.58      1247
    Surprise       0.79      0.83      0.81       831

    accuracy                           0.70      7178
   macro avg       0.69      0.69      0.69      7178
weighted avg       0.70      0.70      0.70      7178
```

Tong so anh du doan sai: 2,125/7,178 (29.6%). Lop **Fear** van la lop yeu nhat (F1=0.55) qua moi lan
thu nghiem - thuong bi nham voi Sad/Angry do dac trung khuon mat chong lan nhieu, kha nang day la gioi
han cua chinh du lieu (noisy labels) hon la cua kien truc model.

### 9.4. Nhan xet ve khoang cach so voi ky vong

Ket qua thuc te (70.40% - 70.77%) thap hon khoang ky vong ban dau (~74-76%) o Muc 6. Nguyen nhan chinh:

1. **Focal Loss chua duoc trien khai** - mot trong cac de xuat co tiem nang cai thien lop hiem (disgust, fear) nhung chua kiem chung.
2. **Overfitting van con nhe o Phase 3** - train acc len den ~77% trong khi val dung o ~70%, cho thay van con du dia de regularize tot hon (vi du Focal Loss, hoac tang data that thay vi chi augment).
3. **Nhieu cau hinh da hoi tu ve cung mot nguong ~70.0-70.6%** du thay doi weight_decay/epoch/Mixup - goi y rang kien truc hien tai (EfficientNet-B0 + CBAM + 3-phase unfreeze) da gan cham tran nang luc voi du lieu va ky thuat hien co; muon vuot qua ro ret can huong di khac han (Focal Loss, kien truc lon hon, hoac them du lieu).
4. Muc ky vong 74-76% o Muc 6 duoc uoc tinh truoc khi thu nghiem thuc te - mang tinh tham khao, khong phai cam ket; so lieu 70.40-70.77% van la cai thien dang ke va hop ly so voi baseline 67.7%.

### 9.5. Files Output (cau hinh moi)

| File | Mo ta |
|---|---|
| model_cnn.pt | Weights Custom CNN (khong doi kien truc) |
| model_mobilenet.pt | Weights EfficientNet-B0 + CBAM (ten file giu nguyen de tuong thich code cu) |
| model_efficientnet_cbam.pt | Ban sao cua model_mobilenet.pt, ten ro rang hon |

---

## 10. Ket Luan Cap Nhat

Sau khi trien khai day du cac de xuat kha thi o Muc 5 (tru Focal Loss), model M2 duoc nang cap tu
**MobileNetV2 (67.7%)** len **EfficientNet-B0 + CBAM (70.40%)**, va **Ensemble M1+M2 dat 70.77%** -
ket qua tot nhat. Gradual Unfreeze va doi kien truc backbone la 2 yeu to dong gop nhieu nhat; Hard
Negative Mining (OHEM) da thu nhung khong hieu qua tren dataset nay, nhieu kha nang do FER-2013 co
nhieu nhan sai (noisy labels) khien viec uu tien sample "loss cao" de bi lan sang sample nham nhan.

Ket qua dat **vuot yeu cau toi thieu cua de bai** (M1 >= 60%, M2 >= 65%) voi bien do an toan, va **Fear**
van la lop kho nhat xuyen suot qua trinh thu nghiem - huong cai thien tiep theo neu can thiet nen uu tien
Focal Loss hoac xu ly rieng cho nhom lop de nham (Fear/Sad/Angry).

---

## 11. Trien Khai & Kiem Thu Ung Dung Local (Inference / Demo)

**Ngay kiem thu:** 05/10/2026

Sau khi hoan thanh training va luu weights, he thong local da duoc dong bo va kiem thu toan dien o moi truong thuc te.

### 11.1. Kien truc mo hinh & Module local
- **`src/model.py`**: Dong bo kien truc `EmotionCNN` va `EmotionEfficientNetCBAM` (kem CBAM Attention), ho tro load linh hoat ca 2 model weights.
- **`src/detector.py`**: Tich hop module phat hien khuon mat YuNet ONNX (`face_detection_yunet_2023mar.onnx`) tuong thich OpenCV 5.0.0, cho toc do va do chinh xac cao hon Haar Cascade truyen thong.
- **`src/predict.py`**: Xu ly chuyen doi anh BGR -> RGB -> Grayscale/ImageNet Transform -> PyTorch Inference -> Softmax confidence.
- **`src/app.py`**: Ung dung chinh ho tro ca 2 che do: **Anh tinh (`run_image`)** va **Luong Realtime (`run_stream`)**.

### 11.2. Thiet lap cau hinh & Toi uu hoa
- **Cai dat `config.yaml`**:
  - Tu dong lay duong dan IP Camera (`url: "http://...:8080/video"`) hoac Webcam local (`--source 0`).
  - Nguong tin cay confidence: `0.4` (40%).
  - Khung hinh toi uu: `640x480`, `target_fps: 20`, `buffer_size: 1` (chong giat lag/tich tu bo dem camera).

### 11.3. Ket qua kiem thu thuc te
1. **Kiem thu Anh tinh (Static Image):**
   - File test: `weights/sample_images.png`
   - YuNet phat hien chinh xac **4/4 khuon mat**.
   - Model `EfficientNet + CBAM` phan loai tot va nhay hon o cac khuon mat kho (dat do tin cay 60.2% va 47.0% so voi CNN bi duoi nguong 40%).
2. **Kiem thu Realtime (Webcam / IP Camera):**
   - Chay thanh cong voi lenh `python src/app.py --model transfer`.
   - Ung dung load model va YuNet ONNX mượt mà, hien thi Bounding Box, nhan cam xuc, phan tram tin cay va chi so FPS realtime theo thoi gian thuc.