from django.shortcuts import render, HttpResponse
from django.core.files.storage import FileSystemStorage
from django.http import FileResponse
import os, datetime
from . import model

# Change the extensions to accept only CSV files
extensions = ['.csv']
is_training = True


def index(request):
    global is_training
    if request.method == 'POST' and request.FILES['data']:
        data_file = request.FILES['data']
        ext = os.path.splitext(data_file.name)[1]
        if ext.lower() in extensions:
            fs = FileSystemStorage()
            filename = fs.save(data_file.name, data_file)
            uploaded_file_url = fs.url(filename)
            file_name = uploaded_file_url.split("/")[2]

            input_path = os.getcwd() + uploaded_file_url
            file_basic_path = os.getcwd() + "/uploads/" + file_name.split(".")[0]
            output_path = os.getcwd() + "/uploads/" + file_name.split(".")[0] + "_processed.json"
            model_path = file_basic_path + "_model.pth"
            print('pwd: ', os.getcwd(), '\n input_path: ', input_path,
                 '\n file_basic_path', file_basic_path, '\n output_path:', output_path )
            print('No. 1: ', is_training)
            if is_training:  # Train the model if it doesn't exist.
                result = model.train(input_path, file_basic_path, output_path)
                uploads_path = uploaded_file_url.split(".")[0]
                model_download_link = '/download_model/' + os.path.basename(model_path)
                is_training = False  # Next time, we will do prediction.
                print('No. 2: ', is_training)
                return render(request, 'HealerML/index.html', {"model_download_link": model_download_link,
                                                               "uploads_path": uploads_path,
                                                               "train_loss": result["train_losses"],
                                                               "val_loss": result["val_losses"],
                                                               "label_counts": result["label_counts"]
                                                               })

            else:  # Load the model and predict if it exists.
                result = model.test(input_path, file_basic_path, output_path)
                model_download_link = '/download_model/' + os.path.basename(model_path)
                download_link = '/download/' + os.path.basename(output_path)
                is_training = True  # Next time, we will do training.
                return render(request, 'HealerML/index.html', {"model_download_link": model_download_link,
                                                               "download_link": download_link,
                                                               "label_counts": result["label_counts"]
                                                               })
        else:
            return HttpResponse("Only Allowed extensions are {}".format(extensions))
    return render(request, 'HealerML/index.html')

# def index(request):
#     if request.method == 'POST' and request.FILES['data']:
#         data_file = request.FILES['data']
#         ext = os.path.splitext(data_file.name)[1]
#         if ext.lower() in extensions:
#             fs = FileSystemStorage()
#             filename = fs.save(data_file.name, data_file)
#             uploaded_file_url = fs.url(filename)
#             file_name = uploaded_file_url.split("/")[2]
#
#             input_path = os.getcwd() + uploaded_file_url
#             basic_path = os.getcwd() + "/uploads/" + file_name.split(".")[0]
#             output_path = os.getcwd() + "/uploads/" + file_name.split(".")[0] + "_processed.csv"
#
#             # remover.process(input_path, output_path)
#             model.preprocess(input_path, input_path)
#             result = model.train(input_path, basic_path, output_path)
#             uploads_path = uploaded_file_url.split(".")[0]
#             data_file_path = uploaded_file_url.split(".")[0] + "_processed.csv"
#             download_link = '/download/' + os.path.basename(output_path)
#             model_download_link = '/download_model/' + os.path.basename(os.getcwd() + "/model/resnet.pt")
#             return render(request, 'HealerML/index.html', {"data_file_path": data_file_path,
#                                                            "download_link": download_link,
#                                                            "model_download_link": model_download_link,
#                                                            "uploads_path": uploads_path,
#                                                            "train_loss": result["train_losses"],
#                                                            "val_loss": result["val_losses"],
#                                                            "label_counts": result["label_counts"]
#                                                            })
#         else:
#             return HttpResponse("Only Allowed extensions are {}".format(extensions))
#     return render(request, 'HealerML/index.html')


def data(request):
    if request.method == 'POST' and request.POST['data']:
        data_file = request.POST['data']
        data_file_name = datetime.datetime.now().strftime("%Y%b%d%H%M%S%f") + ".csv"
        data_file_path = os.getcwd() + "/uploads/" + data_file_name

        with open(data_file_path, "wb") as f:
            f.write(data)

        input_path = os.getcwd() + "/uploads/" + data_file_name
        output_path = os.getcwd() + "/uploads/" + data_file_name.split(".")[0] + "_processed.csv"


        model.preprocess(input_path, input_path)
        result = model.train(input_path, output_path)
        data_file_path = "/uploads/" + data_file_name.split(".")[0] + "_processed.csv"
        return HttpResponse(request.get_host() + data_file_path)


def download(request, file_name):
    file_path = os.getcwd() + "/uploads/" + file_name
    if os.path.exists(file_path):
        response = FileResponse(open(file_path, 'rb'), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="{}"'.format(file_name)
        return response
    else:
        return HttpResponse("Sorry file not found.")


def download_model(request, file_name):
    file_path = os.getcwd() + "/uploads/" + file_name
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=file_name)
    else:
        return HttpResponse("Sorry, model file not found.")
