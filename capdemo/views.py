from django.shortcuts import render

# Create your views here.
from PIL import Image
import torchvision
from io import BytesIO
import os.path as oph
import imquality.brisque as brisque

from .caption_me import cap_example

def index(request):
    context = {}
    return render(request, 'capdemo/index.html', context)


def caption_all(request):

    img_file = request.FILES['fileUpload']
    file_name = img_file.name
    img_bytes = img_file.read()
    x = Image.open(BytesIO(img_bytes)).convert('RGB')
    im_quality = brisque.score(x)
    context = {'im_quality': '{:.2f}'.format(im_quality)}

    if im_quality > 50.0:
        context['no_caption'] = True
    else:
        context['no_caption'] = False 
        x = x.resize((224, 224))
        x = torchvision.transforms.ToTensor()(x)
        x = torchvision.transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])(x)
        context.update(cap_example(x))
        with open(oph.join(oph.dirname(__file__), 'static/capdemo/images/{}'.format(file_name)), 'wb') as f:
            f.write(img_bytes)
    context['filename'] = file_name
    return render(request, 'capdemo/results.html', context)