# Makine öğrenimi

ilk önce MachineLearningModelMaker.py ile model eğitilip kaydediliyor sonra MachineLearningModelImporter.py ile kaydedilen model train ve test datalarıyla çalışıyor. Bunu iki dosyaya ayırma sebebim model test edilirken tekrar eğitilmesine edilmesine gerek olmamasıdır. İlk önce model oluşturulur kaydedilir gerekli testler kaydedilen model üzerinde yapılır.

##Gizli katman adedi ve özellikleri


Multilayer perceptronun bir çeşidi olan Convoluted Neural Networks kullanıldı.Katmanlar 3D dizide (X ekseni koordinatı, Y ekseni koordinatı ve rengi) düzenlendi.

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)

Görüldüğü üzere 8 katman bulunmaktadır modelimizde.

Conv2D katmanı convolutional katmandır, bu katman görüntüdeki özellikleri tanımlayan belirli filtreler uygular.

Maxpooling katmanı pooling katmandır, convolutional katmanlar bir araya getirilerek özelliklerin azaltılmasına yardımcı olur.

Dropout katmanının kullanılma sebebi şudur; nöral ağlar büyük veri kümeleri konusunda eğitilse de aşırı uyum(overfitting) sorunu oluşabilir. Bu sorunu önlemek için eğitim süreci sırasında üniteleri ve bağlantılarını rastgele bırakıyoruz.

Dense katmanının bütün girişleri ve bütün çıkışları birbirine bağlı olan geleneksel katmandır.

##Aktivasyon fonksiyonlarının adı

Aktivasyon fonksiyonlarının adları kodda relu ve softmax olarak geçer. Relu Rectified Linear Units olarak geçer. Sotfmax ise bildiğimiz sotmax aktivasyon fonksiyonu.

##Öğrenme hızı ilk değeri

Öğrenme hızı değerini 00.1 kullandım bu değer optimum sonuçlar veriyordu. 

##Momentum değeri

Bu modelde momentum değerini 0.99 olarak kullandım bunun daha iyi sonuçlar verdiğini elde ettim.

##Decay

Decay’ın model üzerinde daha iyi sonuçlar vermesini sağladığını fark ettiğim için kullandım decay değerini 0.01 olarak kullandım.

##Gradyan yöntemi

Öğrenme hızı ve ağırlık güncellemesindeki gradyan yöntemi SGD olarak kullandım.

##Model eğitilirken öğrenme hızı aşağıdaki gibidir.

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)
![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)
