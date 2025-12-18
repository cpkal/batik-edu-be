class BatikImageService:
    def __init__(self, repository):
        self.repository = repository

    # structure of image_data: {'filename': str, 'path': str, 'metadata': dict}
    def save_image(self, image_data):
        image_id = self.repository.insert(image_data)
        return image_id
      
    def get_all_images(self):
        return self.repository.get_all()