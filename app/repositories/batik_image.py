class BatikImageRepository:
    def __init__(self, collection):
        self.collection = collection

    def insert(self, image_data):
        result = self.collection.insert_one(image_data)
        return str(result.inserted_id)
    
    def get_all(self):
        return list(self.collection.find())