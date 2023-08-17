This submission is divided into 6 parts:

/Models
	Object_Detection.ipynb: Object Detection on Google Colab using YOLOv5s
	New_Model.ipynb: Object Detection on Google Colab using YOLOv8s

/Inference_API: Contains Inference_API.py script responsible for requests like listing models available, listing labels, and uploading files to a specific model to return a json response
	/json: contains objectclasses.json parsed by Inference_API.py
	/templates: contains html files used 
	/models: contains 2 onnx models for yolov5s and yolov8s
	/Uploads: Flask saves uploaded images from the browser in this directory

/results: contains, annotated results from inference api of Yolov5s, and csv files showing metrics across epochs during training.
Note that if you want to run Google colab files, kindly mount drive and upload Dataset.zip (Dataset provided for this project) in MyDrive. You can observe output with TensorBoard.

/Netron: contains images of models represented with Netron

/Data Augmentation
	Data_Augmentation.py: Data Augmentation script
	+ Folders containing zipped results (note that labelling was done with makesense.ai)
