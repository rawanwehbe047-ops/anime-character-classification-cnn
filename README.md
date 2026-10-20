# Anime Character Image Classification with a CNN

**A first deep learning portfolio project by Rawan Wehbe**

This project trains a Convolutional Neural Network (CNN) with PyTorch to classify anime images into two categories: **Anastasia** and **Takao**. It downloads a small image dataset, preprocesses the images, trains the model, plots training/validation loss, and displays predictions.

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rawanwehbe047-ops/anime-character-classification-cnn/blob/main/anime_cnn.ipynb)

## Explore the project

- **For everyone:** read the explanations below and view the training graphs and predictions generated when you run the notebook.
- **For developers:** inspect `anime_cnn.ipynb` or `anime_cnn.py`.
- **To run it interactively:** click the Open in Google Colab badge above, then run the cells from top to bottom.
- **To run locally:** install dependencies with `pip install -r requirements.txt`, then run `python anime_cnn.py`. Internet access is required to download the dataset.

## Project details

| Item | Details |
|---|---|
| Task | Two-class image classification |
| Classes | Anastasia and Takao |
| Dataset | Intended to load 100 images total (50 per class) from the course dataset |
| Split | 80 training images, 20 validation images, if the source dataset contains 100 images |
| Image size | 64 × 64 pixels |
| Model | A small CNN with two convolutional layers and fully connected layers |
| Frameworks/libraries | PyTorch, Torchvision, NumPy, Matplotlib, scikit-learn, Pillow |

## What the notebook demonstrates

1. Downloading and exploring image data.
2. Resizing and normalizing images.
3. Creating a PyTorch Dataset and DataLoaders.
4. Building a CNN with convolution, pooling, and fully connected layers.
5. Training for five epochs and tracking training/validation loss.
6. Viewing validation predictions and calculating validation accuracy.

## Important note about results

The validation set is small, so accuracy can vary between runs and does not prove that the model will perform well on new images. Treat this as a learning project, not a production-ready classifier. Run the notebook to generate the current outputs rather than relying on a saved accuracy number.

## Attribution

This is a personal learning/portfolio adaptation of an IBM Skills Network educational lab. The original lab credits Ricky Shi and includes IBM Corporation copyright notices. This repository focuses on the core experiment and does not include the original lab's exercises or most instructional text. Please refer to the original course material for its complete learning context and licensing terms.

## License

No separate license is provided for this portfolio adaptation. Please confirm that you have the right to license all included material before adding a license.
