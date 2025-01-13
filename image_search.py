import os
import csv
from PIL import Image
from tqdm import tqdm
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FAISS_PATH = './data/vector.idx'


class Load_Data:
    """A class for loading data from single/multiple folders or a CSV file"""

    def __init__(self):
        """
        Initializes an instance of LoadData class
        """
        pass
    
    def from_folder(self, folder_list: list):
        """
        Adds images from the specified folders to the image_list.

        Parameters:
        -----------
        folder_list : list
            A list of paths to the folders containing images to be added to the image_list.
        """
        self.folder_list = folder_list
        image_path = []
        for folder in self.folder_list:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        image_path.append(os.path.join(root, file))
        return image_path

    def from_csv(self, csv_file_path: str, images_column_name: str):
        """
        Adds images from the specified column of a CSV file to the image_list.

        Parameters:
        -----------
        csv_file_path : str
            The path to the CSV file.
        images_column_name : str
            The name of the column containing the paths to the images to be added to the image_list.
        """
        self.csv_file_path = csv_file_path
        self.images_column_name = images_column_name
        image_path = []
        with open(csv_file_path, 'r') as f:
            file = csv.DictReader(f)
            for row in file:
                image_path.append(row[images_column_name])
        return image_path
    

class Search_Setup:
    """ A class for setting up and running image similarity search."""
    def __init__(self, image_list: list, model_name='openai/clip-vit-base-patch16', 
                 image_count: int = None, batch_size: int = 1):
        """
        Parameters:
        -----------
        image_list : list
        A list of images to be indexed and searched.
        model_name : str, optional (default='openai/clip-vit-base-patch16')
        The name of the pre-trained model to use for feature extraction.
        pretrained : bool, optional (default=True)
        Whether to use the pre-trained weights for the chosen model.
        image_count : int, optional (default=None)
        The number of images to be indexed and searched. If None, all images in the image_list will be used.
        """
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name).to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        if image_count==None:
            self.image_list = image_list
        else:
            self.image_list = image_list[:image_count]
        self.batch_size = batch_size
        self.index = None
    
    def _extract(self, img):
        # Preprocess the image
        image_inputs = self.processor(images=img, return_tensors="pt")

        # Extract features
        features = self.model.get_image_features(**image_inputs)
        return features
    
    def _start_feature_extraction(self, image_list):
        i = 0
        n_batch = (len(image_list) + self.batch_size - 1) // self.batch_size
        features = []
        for i in tqdm(range(n_batch)):
            img = [
                Image.open(p).resize((224, 224)).convert('RGB')
                for p in image_list[i*self.batch_size:(i+1)*self.batch_size]
            ]
            features.append(self._extract(img))
        return torch.concat(features, dim=0).detach()
    
    def _start_indexing(self, features, method='ip'):
        d = len(features[0])
        if method == 'l2':
            self.index = faiss.IndexFlatL2(d)
        else:
            self.index = faiss.IndexFlatIP(d)
        self.index.add(features)
        # faiss.write_index(self.index, FAISS_PATH)
        # print(f'Saved the Indexed File at: {FAISS_PATH}')
    
    def run_index(self, method='ip'):
        """
        Indexes the images in the image_list and creates an index file for fast similarity search.
        """
        features = self._start_feature_extraction(self.image_list)
        self._start_indexing(features, method=method)
    
    def add_images_to_index(self, new_image_paths: list):
        """
        Adds new images to the existing index.

        Parameters:
        -----------
        new_image_paths : list
            A list of paths to the new images to be added to the index.
        """
        # Load existing index
        # index = faiss.read_index(FAISS_PATH)

        # Extract features from the new image
        features = self._start_feature_extraction(new_image_paths)

        # Add the new image to the index
        self.image_list.extend(new_image_paths)
        self.index.add(features)

        # Save the updated index
        # faiss.write_index(self.index, FAISS_PATH)
        # print(f'Saved the Indexed File at: {FAISS_PATH}')

    def _search_by_vector(self, v, n: int):
        # index = faiss.read_index(FAISS_PATH)
        D, I = self.index.search(v, n)
        return [
            {'i': I[0][i], 'image': self.image_list[I[0][i]], 'score': D[0][i]}
            for i in range(n)
        ]
    
    def _get_query_text_vector(self, text: str):
        text_inputs = self.processor(text=text, return_tensors="pt")

        # Extract features
        features = self.model.get_text_features(**text_inputs).detach().numpy()
        return features
    
    def _get_query_image_vector(self, text: str):
        image_inputs = self.processor(text=text, return_tensors="pt")

        # Extract features
        features = self.model.get_image_features(**image_inputs).detach.numpy()
        return features
    
    def get_images_by_text(self, text: str, number_of_images: int = 10):
        """
        Returns the most similar images to a given query text according to the indexed image features.

        Parameters:
        -----------
        text : str
            Query text.
        number_of_images : int, optional (default=10)
            The number of most similar images to the query image to be returned.
        """
        features = self._get_query_text_vector(text)
        return self._search_by_vector(features, number_of_images)
    
    def get_images_by_image(self, image_path: str, number_of_images: int = 10):
        """
        Returns the most similar images to a given query image according to the indexed image features.

        Parameters:
        -----------
        image_path : str
            The path to the query image.
        number_of_images : int, optional (default=10)
            The number of most similar images to the query image to be returned.
        """
        features = self._get_query_image_vector(image_path)
        return self._search_by_vector(features, number_of_images)


if __name__ == '__main__':
    # Load images from a folder
    image_list = Load_Data().from_folder(['./data/images'])
    
    # Set up the search engine
    st = Search_Setup(
        image_list=image_list, 
        model_name='openai/clip-vit-base-patch16',
        batch_size=1
    )

    # Index the images
    st.run_index()

    # Search for mage by text
    query = 'shopping for books'
    print(f'Query: {query}')
    print('Results:')
    print(st.get_images_by_text(text=query, number_of_images=2))
