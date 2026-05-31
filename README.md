Reviews Analyzer is a prototype application for analyzing customer reviews. The app can process manually entered text, extract review text from images using OCR, and analyze multiple reviews from a CSV file. For each review, it predicts sentiment, estimates a star rating, detects key aspects, and marks urgent negative feedback.

Requirements

Before running the project, install Tesseract OCR on your computer. It is required for extracting text from images.

After installation, make sure that the path to tesseract.exe is available in the application sidebar or added to your system PATH.

Installation and Run
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py

For macOS / Linux:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py

The training dataset must be located in the data folder and must have the following path:

data/reviews_dataset.csv

The CSV file must contain these columns:

sentiment,stars,review,aspects

Column description:

sentiment — review sentiment label: positive, neutral, or negative
stars — numeric rating from 1 to 5
review — review text used for model training
aspects — review topic or several topics separated by commas, for example support, app, payment, delivery

Example:

sentiment,stars,review,aspects
positive,4,"easy checkout and no weird extra steps no serious issues.",payment
negative,1,"today, support kept asking the same questions",support
neutral,3,"in short, the service request has an identifier status is unchanged.",support
positive,4,"today, not perfect, but overall the service is good helped a lot.",general
negative,1,"from my side, the refund process is a mess I am not happy.","app,payment"

Before training the model, make sure that reviews_dataset.csv exists in the data folder and follows this structure.
