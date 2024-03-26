import matplotlib.pyplot as plt
import matplotlib
from matplotlib.offsetbox import AnchoredText

from skimage import data

matplotlib.rcParams["font.size"] = 10 # rc.Params: yazı boyutunu ayarlar

fig, axes = plt.subplots(1, 2, figsize=(10, 3)) # figsize: resmin boyutunu ayarlar
ax = axes.ravel()

cycle_images = data.stereo_motorcycle() # istediğimiz veriyi kütüphaneden çeker
ax[0].imshow(cycle_images[0]) # 0 ve 1 indeksler, cycle_images adlı bir dizideki belirli görüntülerin indeksleridir. ax[0] ve ax[1], subplotların ilk ve ikinci elemanlarını temsil eder

ax[1].imshow(cycle_images[1])

ax[0].add_artist( 
    AnchoredText( #resmin istediğimiz bölgesinde, yazıda, boyutta yazmamızı sağlar
        "Stereo",
        prop=dict(size=10),
        frameon=True,
        borderpad=0,
        loc="upper left",
    )
)

fig.tight_layout() # Bu yöntem, subplotlar arasındaki boşlukları ve etiketlerin birbirleriyle çakışmasını önler. 

plt.show()

"""
Subplot, bir grafik alanını bölme işlemidir.
 Matplotlib gibi çizim kütüphanelerinde subplotlar, tek bir grafik penceresi (figür) içinde birden fazla grafik 
 veya çizim alanını temsil eder. Subplotlar, bir figürün içinde farklı konumlarda ve boyutlarda birden fazla çizim 
 oluşturmak için kullanılır. Örneğin, bir figürde 2x2 (4) veya 3x3 (9) gibi bir dizi şeklinde subplotlar oluşturabilirsiniz. 
 Her bir subplot, figürün belirli bir bölgesini kaplar ve bu alan içinde çeşitli grafikler veya çizimler oluşturulabilir.

"""

