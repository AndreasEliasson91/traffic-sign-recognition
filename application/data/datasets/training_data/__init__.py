from application.data.models import CustomImageDataset


wikipedia_dataset = CustomImageDataset(
    annotations_file='wikipedia_signs/annotation/annotations_wikipedia_data.csv',
    img_dir='wikipedia_signs/'
)
