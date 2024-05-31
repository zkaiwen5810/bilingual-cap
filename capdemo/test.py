from django.shortcuts import render

# Create your views here.
from PIL import Image
import torchvision
from io import BytesIO
import os.path as oph
from caption_to_anal import cap_example

def index():
    context = {}
    return render(request, 'capdemo/index.html', context)


def caption_all():

    #img_file = request.FILES['fileUpload']
    #file_name = img_file.name
    #img_bytes = img_file.read()
    x = Image.open("/home/suheng/图片/COCO_val2014_000000581717.jpg").convert('RGB')
    x = x.resize((224, 224))
    x = torchvision.transforms.ToTensor()(x)
    x = torchvision.transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])(x)
    contextr,out = cap_example(x)

    return contextr,out
    
    
caption, ans = caption_all()
print(caption)
print(ans.shape)
