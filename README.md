# Anime Character Image Classification with a CNN

**A first deep learning portfolio project by Rawan Wehbe**

This project uses a Convolutional Neural Network (CNN) built with PyTorch to classify anime images into two categories: **Anastasia** and **Takao**. It downloads a small image dataset, preprocesses the images, trains the model, plots training/validation loss, displays predictions, and reports validation accuracy.

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rawanwehbe047-ops/anime-character-classification-cnn/blob/main/anime_cnn.ipynb)

## Explore the project

- **For everyone:** open the notebook on GitHub to read the explanations, or use the Colab button to run it and view the output charts and predictions.
- **For developers:** inspect `anime_cnn.py` and `anime_cnn.ipynb`.
- **To run in Google Colab:** click the badge above and run the notebook cells from top to bottom.
- **To run locally:** install dependencies with `pip install -r requirements.txt`, then run `python anime_cnn.py`. Internet access is required to download the dataset.

## Project details

| Item | Details |
|---|---|
| Task | Two-class image classification |
| Classes | Anastasia and Takao |
| Dataset | Educational course dataset; expected to contain 50 images per class |
| Split | 80% training / 20% validation |
| Image size | 64 × 64 pixels |
| Model | A small CNN with two convolutional layers and fully connected layers |
| Frameworks/libraries | PyTorch, Torchvision, NumPy, Matplotlib, scikit-learn, Pillow |

## What the project demonstrates

1. Downloading and exploring image data.
2. Resizing and normalizing images.
3. Creating a PyTorch Dataset and DataLoaders.
4. Building a CNN with convolution, pooling, and fully connected layers.
5. Training for five epochs and tracking training/validation loss.
6. Viewing validation predictions and calculating validation accuracy.

## Understanding the results

- **Training loss** shows how the model's error on training data changes over time.
- **Validation loss** measures error on images held out during training.
- **Validation accuracy** is the percentage of validation images classified correctly.

The dataset is small, so accuracy can vary and does not prove the model will work well on new images. Treat this as a learning project, not a production-ready classifier. Run the notebook to generate results rather than relying on a fixed accuracy number.

## Attribution

This is a personal learning/portfolio adaptation of an IBM Skills Network educational lab. The original lab credits Ricky Shi and includes IBM Corporation copyright notices. This repository focuses on the core experiment and does not include the original lab's exercises or most instructional text. Please refer to the original course material for its complete learning context and licensing terms.

## License

No separate license is provided for this portfolio adaptation. Please confirm that you have the right to license all included material before adding a license.
