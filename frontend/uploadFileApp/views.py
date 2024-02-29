import torch
from django.http import HttpResponse
from django.shortcuts import render
from transformers import AutoTokenizer

from uploadFileApp import forms
from uploadFileApp.models import (
    APKModel,
    load_apk,
    convert_text_to_features
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Running on {device}')

tokenizer = AutoTokenizer.from_pretrained('elftsdmr/malware-url-detect')
model = APKModel.from_pretrained(
    '/home/black/atbm/checkpoint',
    torch_dtype=torch.float,
    device_map=device,
    args=None
)


def model_form_upload(request):
    if request.method == 'POST':
        form = forms.FileUploadModelForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.save()
            file_path = '/home/black/atbm/frontend' + file.file.url

            extracted_data = load_apk(file_path)

            input_ids, attention_mask = convert_text_to_features(
                text=extracted_data,
                tokenizer=tokenizer
            )

            input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=device)

            with torch.no_grad():
                inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }

                logits = model(**inputs)

                label = torch.argmax(logits).tolist()

            if label == 1:
                return render(request, 'rejected.html')
            else:
                return render(request, 'accepted.html')

    else:
        form = forms.FileUploadModelForm()

    return render(request, 'upload_form.html', {'form': form})


def loader(request):
    return HttpResponse("")
