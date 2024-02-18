from django import forms

from uploadFileApp.models import File


class FileUploadModelForm(forms.ModelForm):
    class Meta:
        model = File
        fields = ('file',)

        widgets = {
            'file': forms.ClearableFileInput(attrs={'class': 'form-control'}),
        }

    def clean_file(self):
        file = self.cleaned_data['file']
        ext = file.name.split('.')[-1].lower()
        if ext not in ['apk']:
            raise forms.ValidationError("File type not!")
        return file
