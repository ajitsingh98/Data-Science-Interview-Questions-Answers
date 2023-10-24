# Deep Learning Interview Questions


Topics
---

- [Convolution Neural Networks](#convolution-neural-networks)


## Convolution Neural Networks

Contents
----

1. [CNN as Fixed Feature Extractor](#CNN-as-Fixed-Feature-Extractor)
2. [Fine-tuning CNNs](#Fine-Tuning-CNNs)
3. [Neural Style Transfer](#Neural-Style-Transfer)

### CNN as Fixed Feature Extractor

1.  **True or False**: While AlexNet used 11 × 11 sized filters, the main novelty presented in the VGG architecture was utilizing filters with much smaller spatial extent, sized $3 × 3$.

---

2. **True or False**: Unlike CNN architectures such as AlexNet or VGG, ResNet does not
have any hidden FC layers?

---

3. Assuming the VGG-Net has 138, 357, 544 floating point parameters, what is the physical size in Mega-Bytes (MB) required for persisting a trained instance of VGG-Net on permanent storage?

---

4. **True or False**: Most attempts at researching image representation using FE, focused
solely on reusing the activations obtained from layers close to the output of the CNN, and more specifically the fully-connected layers.

---

5.  **True or False**: FE in the context of deep learning is particularly useful when the target
problem does not include enough labeled data to successfully train CNN that generalizes well.

---

6.  Why is a CNN trained on the ImageNet dataset a good candidate for a source problem?

---

7. Complete the missing parts regarding the VGG19 CNN architecture: 

        1. The VGG19 CNN consists of [...] layers.
        2. It consists of [...] convolutional and 3 [...] layers.
        3. The input image size is [...].
        4. The number of input channels is [...].
        5. Every image has it’s mean RGB value [subtracted / added].
        6. Each convolutional layer has a [small/large] kernel sized [...].
        7. The number of pixels for padding and stride is [...].
        8. There are 5 [...] layers having a kernel size of [...] and a stride of [...] pixels. 9. For non-linearity a [rectified linear unit (ReLU [5])/sigmoid] is used.
        10. The [...] FC layers are part of the linear classifier.
        11. The first two FC layers consist of [...] features.
        12. The last FC layer has only [...] features.
        13. The last FC layer is terminated by a [...] activation layer. 
        14. Dropout [is / is not] being used between the FC layers.

---

8. The following question discusses the method of fixed feature extraction from layers of the
VGG19 architecture for the classification of pancreatic cancer. It depicts FE principles which are applicable with minor modifications to other CNNs as well. Therefore, if you happen to encounter a similar question in a job interview, you are likely be able to cope with it by utilizing the same logic. In Fig. (9.7) three different classes of pancreatic cancer are displayed: A, B and C, curated from a dataset of 4K Whole Slide Images (WSI) labeled by a board certified pathologist. Your task is to use FE to correctly classify the images in the dataset.
<table align='center'>
  <tr>
    <td align="center">
      <img src="img/cnn_feature-1.png" alt= "A dataset of 4K histopathology WSI from three severity classes: A, B and C" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center">A dataset of 4K histopathology WSI from three severity classes: A, B and C </td>
  </tr>
</table>
Table (9.3) presents an incomplete listing of the of the VGG19 architecture. As depicted, for each layer the number of filters (i. e., neurons with unique set of parameters), learnable parameters (weights,biases), and FV size are presented.
<table align='center'>
  <tr>
    <td align="center">
      <img src="img/cnn_feature-2.png" alt= "Incomplete listing of the VGG19 architecture" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center">Incomplete listing of the VGG19 architecture </td>
  </tr>
</table>

1. Describe how the VGG19 CNN may be used as fixed FE for a classification task. In your answer be as detailed as possible regarding the stages of FE and the method used for classification.

2. Referring to Table (9.3), suggest three different ways in which features can be extrac- ted from a trained VGG19 CNN model. In each case, state the extracted feature layer name and the size of the resulting FE.

3. After successfully extracting the features for the 4K images from the dataset, how can you now classify the images into their respective categories?

---

9. Still referring to Table (9.3), a data scientist suggests using the output layer of the
VGG19 CNN as a fixed FE. What is the main advantage of using this layer over using for instance, the fc7 layer? (Hint: think about an ensemble of feature extractors)

---

10. Still referring to Table (9.3) and also to the code snippet in Fig.(7.4), which represents a
new CNN derived from the VGG19 CNN:

        ```python
        import torchvision.models as models
        ...
        class VGG19FE(torch.nn.Module):
        def __init__(self):
        super(VGG19FE, self).__init__()
        original_model = models.VGG19(pretrained=[???])
        self.real_name = (((type(original_model).__name__)))
        self.real_name = "vgg19"
        self.features = [???]
        self.classifier = torch.nn.Sequential([???])
        self.num_feats = [???]
        def forward(self, x):
            f = self.features(x)
            f = f.view(f.size(0), -1)
            f = [???]
            print (f.data.size())
            return f
        ```
1. Complete line 6; what should be the value of `pretrained` ?
2. Complete line 10; what should be the value of `self.features` ?
3. Complete line 12; what should be the value of `self.num_feats` ? 
4. Complete line 17; what should be the value of `f` ?

---

11. We are still referring to Table (9.3) and using the skeleton code provided in Fig. (7.5)
to derive a new CNN entitled `ResNetBottom` from the ResNet34 CNN, to extract a 512- dimensional FV for a given input image. 
Complete the code as follows:

```python

import torchvision.models as models
res_model = models.resnet34(pretrained=True)
class ResNetBottom(torch.nn.Module):
 def __init__(self, original_model):
 super(ResNetBottom, self).__init__()
 self.features = [???]
 def forward(self, x):
    x = [???]
    x = x.view(x.size(0), -1)
    return x
    
```

1. The value of self.features in line 7. 
2. The forward method in line 11.

---

12. Still referring to Table (9.3), the PyTorch based pseudo code snippet in Fig. (7.6) returns
the 512-dimensional FV from the modified ResNet34 CNN, given a 3-channel RGB image as an input.

```python

import torchvision.models as models
from torchvision import transforms
...
test_trans = transforms.Compose([
transforms.Resize(imgnet_size),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406],
[0.229, 0.224, 0.225])])

def ResNet34FE(image, model):
    f=None
    image = test_trans(image)
    image = Variable(image, requires_grad=False).cuda()
    image= image.cuda()
    f = model(image)
    f = f.view(f.size(1), -1)
    print ("Size : {}".format(f.shape))
    f = f.view(f.size(1),-1)
    print ("Size : {}".format(f.shape))
    f =f.cpu().detach().numpy()[0]
    print ("Size : {}".format(f.shape))
    return f

```

*PyTorch code skeleton for extracting a 512-dimensional FV from a pre-trained ResNet34 CNN model.*

Answer the following questions regarding the code in Fig. (7.6):

1. What is the purpose of `test_trans` in line 5?
2. Why is the parameter `requires_grad` set to False in line 14? 3. What is the purpose of `f.cpu()` in line 23?
4. What is the purpose of `detach()` in line 23?
5. What is the purpose of `numpy()[0]` in line 23?

---

13. Define the term fine-tuning (FT) of an ImageNet pre-trained CNN.

---

14. Describe three different methods by which one can fine-tune an ImageNet pre-trained CNN.

---

15. Melanoma is a lethal form of malignant skin cancer, frequently misdiagnosed as a benign
skin lesion or even left completely undiagnosed.
In the United States alone, melanoma accounts for an estimated 6, 750 deaths per annum. With a 5-year survival rate of 98%, early diagnosis and treatment is now more likely and possibly the most suitable means for melanoma related death reduction. Dermoscopy images, shown in Fig. (7.7) are widely used in the detection and diagnosis of skin lesions. Dermatologists, relying on personal experience, are involved in a laborious task of manually searching dermoscopy images for lesions.

Therefore, there is a very real need for automated analysis tools, providing assistance to clinicians screening for skin metastases. In this question, you are tasked with addressing some of the fundamental issues DL researchers face when building deep learning pipelines. As suggested in, you are going to use ImageNet pre-trained CNN to resolve a classification task.
<table align='center'>
  <tr>
    <td align="center">
      <img src="img/cnn_feature-3.png" alt= "Skin lesion categories. An exemplary visualization of melanoma" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center">Skin lesion categories. An exemplary visualization of melanoma</td>
  </tr>
</table>

1. Given that the skin lesions fall into seven distinct categories, and you are training using cross-entropy loss, how should the classes be represented so that a typical PyTorch training loop will successfully converge?
2. Suggest several data augmentation techniques to augment the data.
3. Write a code snippet in PyTorch to adapt the CNN so that it can predict 7 classes instead of the original source size of 1000.
4. In order to fine tune our CNN, the (original) output layer with $1000$ classes was removed and the CNN was adjusted so that the (new) classification layer comprised seven softmax neurons emitting posterior probabilities of class membership for each lesion type.

---

### Neural Style Transfer

16. Briefly describe how neural style transfer (NST) works?

---

17.  Complete the sentence: When using the VGG-19 CNN for neural-style transfer, there different images are involved. Namely they are: `[...]`, `[...]` and `[...]`

---

18. Refer to below figure and answer the following questions:

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/nst-1.png" alt= "Artistic style transfer using the style of Francis Picabia’s Udnie painting" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center">Artistic style transfer using the style of Francis Picabia’s Udnie painting</td>
  </tr>
</table>

1. Which loss is being utilized during the training process?
2. Briefly describe the use of activations in the training process.

---

19. Still referring to above image:
    1. How are the activations utilized in comparing the content of the content image to the content of the combined image?
    2. How are the activations utilized in comparing the style of the content image to the style of the combined image?

---

20. Still referring to the image in **Q18**. For a new style transfer algorithm, a data scientist extracts a
feature vector from an image using a pre-trained `ResNet34 CNN`.

```python

import torchvision.models as models
...
res_model = models.resnet34(pretrained=True)

```

He then defines the cosine similarity between two vectors:

$u = {u_1,u_2,....,u_N}$
$v = {v_1,v_2,....,v_N}$

as:

$$
sim(u, v) = \frac{u.v}{|u||v|} = \frac{\sum_{i=1}^Nu_iv_i}{\sqrt{(\sum_{i=1}^Nu_i^2)(\sum_{i=1}^Nv_i^2)}}
$$

Thus, the cosine similarity between two vectors measures the cosine of the angle between the vectors irrespective of their magnitude. It is calculated as the dot product of two numeric vectors, and is normalized by the product of the length of the vectors.

Answer the following questions:
1. Define the term *Gram matrix*.
2. Explain in detail how vector similarity is utilised in the calculation of the Gram matrix during the training of *NST*.

---