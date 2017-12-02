# Image BMI Predictor
---

## Environment Setup

	conda create --name bmi python=3.4

	source activate bmi

	conda install scikit-learn==0.17.1


## How to run?

1. Make Executable

	chmod +x image_BMI_predictor.py

2. Copy image to be tested into INPUT folder

	cp /path/to/input/image.jpeg INPUT/

3. Run image_BMI_predictor.py with test image as argument

	./image_BMI_predictor.py image.jpeg

4. View Output

	cat img_BMI_prediction.csv
