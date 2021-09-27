# Image Scorer

Pytorch implementation of an aesthetic and technical image quality model based on Google's research
paper ["NIMA: Neural Image Assessment"](https://arxiv.org/pdf/1709.05424.pdf). You can find a quick introduction on
their [Research Blog](https://research.googleblog.com/2017/12/introducing-nima-neural-image-assessment.html).

For more information on models:

* NVIDIA Developer
  Blog: [Deep Learning for Classifying Hotel Aesthetics Photos](https://devblogs.nvidia.com/deep-learning-hotel-aesthetics-photos/)
*
Medium: [Using Deep Learning to automatically rank millions of hotel images](https://medium.com/idealo-tech-blog/using-deep-learning-to-automatically-rank-millions-of-hotel-images-c7e2d2e5cae2)

Original tensorflow
version: [https://idealo.github.io/image-quality-assessment/](https://idealo.github.io/image-quality-assessment/).

## Run server

    ```bash
     uvicorn server:app --reload
    ```
