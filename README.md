# Phase-Preserving-Image-Denoising
Python implementation of Peter Kovesi's Phase Preserving Denoising of Images[[1]](#1).
This script is completely based on [Peter Kovesi's Matlab script](https://www.peterkovesi.com/matlabfns/index.html#noisecomp), refer original author's [website](https://www.peterkovesi.com/index.html) for more information.

## Demo:
```python
from ppdenoise import ppdenoise
from skimage import io, img_as_float
import matplotlib.pyplot as plt

img = img_as_float(io.imread("sample.jpg", as_gray=True))
img_denoised = ppdenoise(img)

plt.figure()
plt.subplot(121)
plt.title("original image")
plt.imshow(img, cmap="gray")
plt.subplot(122)
plt.title("Denoised image")
plt.imshow(img_denoised, cmap="gray")
plt.show()
```
## Result:
![Figure_1](https://user-images.githubusercontent.com/39316548/124165855-e4f80400-dabf-11eb-816f-0e17e261f369.png)
## Reference
<a id="1">[1]</a>
[Peter Kovesi, "Phase Preserving Denoising of Images". 
The Australian Pattern Recognition Society Conference: DICTA'99. 
December 1999. Perth WA. pp 212-217](https://www.peterkovesi.com/papers/denoise.pdf)
<br>
