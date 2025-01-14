# Image_Search

Django app to search for images by text.

<img src="screenshots/screenshot_1.jpg" alt="App screenshot" width="561">

The overall workflow is illustrated below. Search results are ranked by similarity between text and image embeddings. Top images are evaluated by asking and answering questions about it with the power of large language models and visual question answering models.

<img src="screenshots/screenshot_2.png" alt="Workflow" width="561">

### Installation
To run the app locally, install the libraries
```
pip install -r requirements.txt
```

### Run
Run migrations
```
python manage.py migrate
```

Start the development server
```
python manage.py runserver
```

### Functionality
- Users can upload multiple images.
- They can search through the uploaded images using text queries.
- They can specify how many top results to show.
- Results are displayed in a grid with their similarity scores and matched attributes.
