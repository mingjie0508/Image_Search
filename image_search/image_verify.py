import os
import json
from huggingface_hub import InferenceClient


HF_API_KEY = os.environ['HUGGINGFACEHUB_KEY']


class Search_Vertify:
    def __init__(self, 
                 parse_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", 
                 vqa_model_name: str = "dandelin/vilt-b32-finetuned-vqa"
        ):
        self.parse_model_name = parse_model_name
        self.vqa_model_name = vqa_model_name
    
    def parse_text(self, text: str):
        client = InferenceClient(api_key=HF_API_KEY)
        query = """Return your answer in JSON format. {"entities": [{"name": "", "type": ""}, {"name": "", "type": ""}]} Please extract entities from the image description: %s.""" % text
        messages = [{"role": "user", "content": query}]
        completion = client.chat.completions.create(
            model=self.parse_model_name, 
            messages=messages, 
            max_tokens=500,
            temperature=0.5
        )
        entities = completion.choices[0].message.content
        try:
            return json.loads(entities)["entities"]
        except:
            return []
    
    def verify_image(self, image_path: str, entities: str):
        matched = []
        discriminator = "yes"
        client = InferenceClient(api_key=HF_API_KEY)
        for ent in entities:
            e = ent["name"]
            query = f"""Is there {e} in the image?"""
            completion = client.visual_question_answering(
                image=image_path,
                question=query,
                model=self.vqa_model_name
            )
            if completion[0].answer == discriminator:
                matched.append({"name": e, "score": completion[0].score})
            else:
                matched.append({"name": e, "score": 0.0})
        return matched

    def verify(self, image_path: str, query: str):
        entities = self.parse_text(query)
        matched = self.verify_image(image_path, entities)
        return matched
    
    def match(self, image_path: str, query: str, threshold: float = 0.5):
        entities = self.parse_text(query)
        matched = self.verify_image(image_path, entities)
        matched = [m["name"] for m in matched if m['score'] > threshold]
        return matched


if __name__ == '__main__':
    image_path = "./data/images/n02823428_beer_bottle.JPEG"
    query = "five bottles"

    sv = Search_Vertify()
    matched = sv.verify(image_path, query)

    print(f"Image: {image_path}")
    print(f"Query: {query}")
    print("Results:")
    print(matched)
