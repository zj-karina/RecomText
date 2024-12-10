# RecomText - Multimodal Recommendation System

A recommendation system that combines textual descriptions and demographic data to create personalized recommendations.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/RecomText.git
   cd RecomText
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # for Linux/Mac
   venv\Scripts\activate  # for Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

1. Place your source data in the `data/` directory:
   - train_events.csv
   - video_info_v2.csv
   - train_targets.csv
   - all_events.csv

2. Run the data preparation script:
   ```bash
   python -m data.baseline_socdem
   ```

## Training

1. Configure parameters in `configs/config.yaml`

2. Start training via Jupyter notebook:
   ```bash
   jupyter notebook notebooks/train.ipynb
   ```

   Alternatively, use the training script:
   ```bash
   python train.py
   ```

## Inference

Use the following code for getting recommendations:

допишу позже

## Project Structure

- `data/` - data processing modules
- `models/` - model architectures
- `utils/` - utility functions
- `notebooks/` - jupyter notebooks
- `configs/` - configuration files

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (optional)

## Features

- Multimodal embeddings combining text and IDs
- Contrastive learning approach
- Efficient batch processing
- Support for both CPU and GPU inference
- Configurable model architecture

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
