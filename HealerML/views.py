# import necessary libraries
from django.shortcuts import render, HttpResponse
from django.core.files.storage import FileSystemStorage
from django.http import FileResponse
import os
import datetime
from . import model

# define global variables
# list of allowed extensions for the data file
extensions = ['.csv']
# boolean variable to check if it's the training phase or not
is_training = True
# path to the saved model
model_path = ''


# function to handle the main page of the web application
def index(request):
    # use global variables declared above
    global is_training, model_path

    # if the request method is POST and there is a file attached
    if request.method == 'POST' and request.FILES['data']:
        data_file = request.FILES['data']
        ext = os.path.splitext(data_file.name)[1]

        # if the file has a correct extension
        if ext.lower() in extensions:
            fs = FileSystemStorage()
            filename = fs.save(data_file.name, data_file)
            uploaded_file_url = fs.url(filename)
            file_name = uploaded_file_url.split("/")[2]
            file_basic_path = os.getcwd() + "/uploads/" + file_name.split(".")[0]

            # if it's the training phase
            if is_training:
                result = model.train(file_basic_path)
                uploads_path = uploaded_file_url.split(".")[0]
                model_path = file_basic_path + "_model.pth"
                model_download_link = '/download_model/' + os.path.basename(model_path)
                is_training = False
                return render(request, 'HealerML/index.html', {
                    "model_download_link": model_download_link,
                    "uploads_path": uploads_path,
                    "file_basic_path": file_basic_path,
                    "train_loss": result["train_losses"],
                    "val_loss": result["val_losses"],
                    "label_counts": result["label_counts"]
                })

            # if it's the testing phase
            else:
                result = model.test(file_basic_path)
                uploads_path = uploaded_file_url.split(".")[0]
                model_download_link = '/download_model/' + os.path.basename(model_path)
                download_link = '/download/' + os.path.basename(file_basic_path + '_predicted.json')
                is_training = True
                return render(request, 'HealerML/index.html', {
                    "model_download_link": model_download_link,
                    "uploads_path": uploads_path,
                    "file_basic_path": file_basic_path,
                    "download_link": download_link,
                    "label_counts": result["label_counts"]
                })

        else:
            return HttpResponse("Only Allowed extensions are {}".format(extensions))

    # if the request method is not POST or there is no file attached
    return render(request, 'HealerML/index.html')


# function to handle the data endpoint of the web application
def data(request):
    # if the request method is POST and there is some data attached
    if request.method == 'POST' and request.POST['data']:
        data_file_name = datetime.datetime.now().strftime("%Y%b%d%H%M%S%f") + ".csv"
        data_file_path = os.getcwd() + "/uploads/" + data_file_name

        # write the data to a file
        with open(data_file_path, "wb") as f:
            f.write(data)
        data_file_path = "/uploads/" + data_file_name.split(".")[0] + "_predicted.csv"

        # return the path to the file
        return HttpResponse(request.get_host() + data_file_path)


# function to handle the download endpoint of the web application
def download(request, file_name):
    file_path = os.getcwd() + "/uploads/" + file_name

    # if the file exists, return it as a response
    if os.path.exists(file_path):
        response = FileResponse(open(file_path, 'rb'), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="{}"'.format(file_name)
        return response

    # if the file doesn't exist, return an error message
    else:
        return HttpResponse("Sorry file not found.")


# function to handle the download_model endpoint of the web application
def download_model(request, file_name):
    file_path = os.getcwd() + "/uploads/" + file_name

    # if the model file exists, return it as a response
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=file_name)

    # if the model file doesn't exist, return an error message
    else:
        return HttpResponse("Sorry, model file not found.")
