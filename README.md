# T-Brain_Orchid
2022 T-Brain競賽 [尋找花中君子 - 蘭花種類辨識及分類競賽](https://tbrain.trendmicro.com.tw/Competitions/Details/20)

## 說明
此競賽目的為圖片分類蘭花共219種，訓練集2190張(每種各10張)，預測集8萬多張。

## 紀錄
1. 使用[EfficientNetV2](https://github.com/leondgarse/keras_efficientnet_v2)之模型架構(效果最佳)，並於尾部加上全連階層&dropout，最終達到驗證集90%左右的正確率。
2. 使用noisy student進行半監督學習，使用爬蟲的蘭花圖片約2萬張(unlabel)。

## 結果
名次/參賽隊伍
正確率:88.5%

## 心得
1. 使用SOTA的預訓練模型效果出乎預料的好，並且由於全凍結(只訓練多加的尾巴)，訓練速度很快，並且模型有很好的穩健性
2. noisy student於此case的效果不佳，推測原因為資料增量的蘭花照片很難平衡(特殊品種幾乎沒找到更多圖片)，但還是將此做法代碼保留。

## 資料夾&代碼說明
* train : 訓練模型用
  * EfficientNetV2_model.py : 建構模型
  * loss.py : 用於metrics計算F1 loss
  * train_EfficientNet_model.py : 訓練模型
  * retrain_aug_EfficientNet_model.py : 訓練模型(包含random_aug.py的資料增強)
  * random_aug.py : 使用imgaug套件進行資料增強
  * resize.py : 將所有圖片轉為640*640(不足的補黑邊)

* predict : 預測結果用
  * predict.py : 預測預測集之結果

* student_train : 訓練noisy student模型用
  * gene_img_csv.py : 將unlabel圖片產生csv檔


## 執行方式
