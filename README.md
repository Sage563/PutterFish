# PutterFish
<p align="center">
  <img src="icon.ico" alt="Image" />
</p>
PutterFish is a chess engine that uses deep learning to evaluate chess positions and make decisions. It is designed to be fast and efficient, making it suitable for use in real-time applications such as online chess platforms and chess analysis tools.

## Features
- Deep learning-based evaluation function
- Fast move generation
- Support for standard chess rules and variants
- Open-source and customizable

## Installation
To install PutterFish, follow these steps:
1. Clone the repository:
    ```git clone github.com/Sage563/PutterFish.git```
2. Install the required dependencies:
    ```pip install -r requirements.txt```
3. Run the engine:
    ```python main.py```
## Usage
PutterFish can be used in various ways, including:
- As a standalone chess engine for analysis and play
- Integrated into online chess platforms for real-time play
- Used in chess analysis tools to evaluate positions and suggest moves
- Uses Stockfish as a baseline for training and evaluation and can be used to compare the performance of PutterFish against Stockfish. Also is a UCI engine, so it can be used with any chess GUI that supports UCI engines.

## Contributing
Contributions to PutterFish are welcome! If you have an idea for a new feature or improvement, please open an issue or submit a pull request.

## License
PutterFish is licensed under the GNU General Public License v3.0  License. See the LICENSE file for more information.

### Traning
To help train and contrinube to the development of PutterFish, use the current Model. To generate a usabe dataset, use the command 
```python
python tranning/dataset.py .\elite -o dataset.json -d 14 -s stockfish-18.exe -t 16 --hash 2048 --workers 12 --batch-size 16 --every-move
```
This command will generate a dataset of chess positions and their evaluations using Stockfish as the evaluation engine. The dataset will be saved in a file called `dataset.json`. The `-d` flag specifies the depth of the search, the `-s` flag specifies the path to the Stockfish executable, the `-t` flag specifies the number of threads to use, the `--hash` flag specifies the size of the hash table, the `--workers` flag specifies the number of worker processes to use, the `--batch-size` flag specifies the batch size for processing positions, and the `--every-move` flag indicates that evaluations should be generated for every move in the game. The `.\elite` directory should contain the PGN files of chess games that you want to use for training. You can customize the command with different parameters to suit your needs.

### Training model with DATASET

To train the PutterFish model using the generated dataset, you can use the following command:
```python
python tranning/train_model.py --dataset dataset.json --model chess_model.pth --output chess_model.pth --epochs 20 --batch-size 64 --learning-rate 0.0001 --device cuda  
```

This command will train the model using the dataset specified in `dataset.json`, useses a pretrained model `chess_model.pth`, and use the specified parameters for training. The `--epochs` flag specifies the number of training epochs, the `--batch-size` flag specifies the batch size for training, the `--learning-rate` flag specifies the learning rate for the optimizer, and the `--device` flag specifies the device to train on (e.g., "cuda" for GPU or "cpu" for CPU). You can adjust these parameters as needed to achieve better performance.

To get the pre-traind model to help you get started, you can download it from the following link: [model.pth](model.pth). This model has been trained on a large dataset of chess positions and can be used as a starting point for further training or fine-tuning. You can use this pre-trained model to evaluate chess positions and make decisions in your own applications, or you can use it as a baseline for training your own model with additional data or different parameters. To use the pre-trained model, simply specify the path to the model file in the `--model` flag when running the training command. For example:
This will load the pre-trained model from `model.pth` and use it as the starting point for training. You can then fine-tune the model with your own dataset and parameters to improve its performance.

## Contribution using tranined model

To give us the model and contribute to the development of PutterFish, you can follow these steps:
1. Train your model using the command mentioned above, making sure to save the trained model to a file (e.g., `chess_model.pth`).
2. Once you have trained your model, you can share it with us by uploading the model file to a file-sharing service (e.g., Google Drive, Dropbox) and providing us with the download link.
3. Alternatively, you can submit a pull request to the PutterFish repository with your trained model file included. Make sure to include a description of the training process, the dataset used, and any relevant parameters or settings that you used during training.
4. We will review your contribution and, if it meets our standards, we will merge it into the main branch of the repository. Your contribution will help improve the performance of PutterFish and make it more effective for evaluating chess positions and making decisions. Thank you for your contribution!

Have fun and I LIKE FISH!


### Credits
- The PutterFish project was developed by Sage563.
- Icon made by @Sana
- The project uses the Stockfish chess engine for training and evaluation.
- The project is open-source and welcomes contributions from the community.
- First set of games used for training were taken from the Elite database, which is a collection of high-quality chess games. [Elite Database](https://database.nikonoel.fr/) 3gb of pgns

